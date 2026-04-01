import os
from flask import Flask, request, jsonify, render_template_string, session
from flask_cors import CORS
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv
import json
import sqlite3
from datetime import datetime, date, timedelta
import hashlib
import re
import base64
import io
from gtts import gTTS
import pygame
import tempfile
import threading
import time
import random
import numpy as np

# --- Configuration and Initialization ---

load_dotenv()

app = Flask(__name__)
CORS(app)
app.secret_key = 'ai-study-suite-pro-secret-key-2024'

# Increase file size limit (100MB)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Initialize pygame mixer for audio playback
if not pygame.mixer.get_init():
    try:
        pygame.mixer.init()
        print("🔊 Pygame Mixer initialized.")
    except Exception as e:
        print(f"❌ Pygame Mixer failed to initialize: {e}")

# Database setup
def init_db():
    conn = sqlite3.connect('study_history.db')
    c = conn.cursor()
    
    # Study sessions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS study_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_hash TEXT UNIQUE,
            filename TEXT,
            content_preview TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            key_concepts_count INTEGER,
            flashcards_count INTEGER,
            mcqs_count INTEGER,
            full_content TEXT
        )
    ''')
    
    # Chat history table
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_hash TEXT UNIQUE,
            filename TEXT,
            extracted_formulas TEXT,
            chat_history TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # MCQ performance table
    c.execute('''
        CREATE TABLE IF NOT EXISTS mcq_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_hash TEXT,
            total_questions INTEGER,
            correct_answers INTEGER,
            score_percentage REAL,
            weak_areas TEXT,
            strong_areas TEXT,
            detailed_analysis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # User settings table
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            theme TEXT DEFAULT 'dark',
            ai_model TEXT DEFAULT 'gemini-2.5-flash',
            voice_enabled INTEGER DEFAULT 1,
            voice_speed REAL DEFAULT 1.0,
            voice_gender TEXT DEFAULT 'female',
            voice_language TEXT DEFAULT 'en',
            mcq_count INTEGER DEFAULT 35,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Daily revision tests table
    c.execute('''
        CREATE TABLE IF NOT EXISTS daily_revision_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_date TEXT UNIQUE,
            questions TEXT,
            completed INTEGER DEFAULT 0,
            score REAL DEFAULT 0,
            analysis_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Test scores table for timer functionality
    c.execute('''
        CREATE TABLE IF NOT EXISTS test_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_date TEXT,
            score REAL,
            test_type TEXT,
            duration INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Goals/To-do list table
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            category TEXT,
            priority TEXT,
            due_date TEXT,
            completed INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Focus sessions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS focus_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            duration INTEGER,
            session_type TEXT,
            completed INTEGER DEFAULT 0,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Test sessions table for enhanced testing system
    c.execute('''
        CREATE TABLE IF NOT EXISTS test_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_hash TEXT,
            test_type TEXT,
            questions TEXT,
            user_answers TEXT,
            score REAL,
            time_taken INTEGER,
            completed INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check for initial settings entry
    c.execute('SELECT COUNT(*) FROM user_settings')
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO user_settings (theme, ai_model, voice_enabled, voice_speed, voice_gender, voice_language, mcq_count) VALUES ('dark', 'gemini-2.5-flash', 1, 1.0, 'female', 'en', 35)")
    
    # Check for today's revision test
    today = date.today().isoformat()
    c.execute('SELECT COUNT(*) FROM daily_revision_tests WHERE test_date = ?', (today,))
    if c.fetchone()[0] == 0:
        c.execute('INSERT INTO daily_revision_tests (test_date) VALUES (?)', (today,))

    conn.commit()
    conn.close()

init_db()

# Configure the Gemini API client
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ GEMINI_API_KEY not found in environment variables. AI features disabled.")
        model = None
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("✅ Gemini API configured successfully")
except Exception as e:
    print(f"❌ Error configuring Generative AI: {e}")
    model = None

# Global variable for audio control
current_audio = None
audio_playing = False

# Timer constants
DEFAULT_TEST_DURATION = 30  # minutes

# --- Core Functions ---

def extract_text_from_pdf(file):
    """Extracts text from a PDF file with large file support."""
    try:
        file.seek(0)
        reader = PyPDF2.PdfReader(file)
        text = ""
        for i, page in enumerate(reader.pages):
            if i >= 50:
                break
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
            if len(text) > 100000:
                break
        print(f"📄 Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return ""

def extract_formulas_from_text(text):
    """Extract mathematical formulas and equations from text."""
    try:
        patterns = [
            r'\$[^$]+\$',
            r'\\\[.*?\\\]',
            r'\\\(.*?\\\)',
            r'[A-Za-z]+\s*=\s*[^;\n]+',
            r'[A-Za-z]+\s*:\s*[^;\n]+',
            r'\b(sin|cos|tan|log|ln|exp|sqrt|sum|integral|derivative)\s*\([^)]+\)',
            r'[A-Za-z]_[0-9]',
            r'[A-Za-z]\^[0-9]',
            r'[A-Za-z]+\s*=\s*[^=]+?(?=\n\n|\n[A-Z]|$)',
        ]
        
        formulas = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            formulas.extend(matches)
        
        unique_formulas = list(set(formulas))
        cleaned_formulas = [formula.strip() for formula in unique_formulas if len(formula.strip()) > 2]
        
        print(f"🔍 Extracted {len(cleaned_formulas)} formulas")
        return cleaned_formulas[:50]
    except Exception as e:
        print(f"❌ Error extracting formulas: {e}")
        return []

def save_study_session(filename, content_preview, content):
    """Save study session to history database."""
    try:
        session_hash = hashlib.md5(f"{filename}{content_preview}{datetime.now()}".encode()).hexdigest()[:10]
        full_content_json = json.dumps(content)
        
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO study_sessions
            (session_hash, filename, content_preview, key_concepts_count, flashcards_count, mcqs_count, full_content)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_hash,
            filename,
            content_preview[:200],
            len(content.get('key_concepts', [])),
            len(content.get('flashcards', [])),
            len(content.get('mcqs', [])),
            full_content_json
        ))
        conn.commit()
        conn.close()
        print(f"✅ Study session for {filename} saved successfully.")
        return session_hash
    except Exception as e:
        print(f"❌ Error saving study session: {e}")
        return None

def save_mcq_performance(session_hash, total_questions, correct_answers, weak_areas, strong_areas, detailed_analysis):
    """Save MCQ performance to database."""
    try:
        score_percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        weak_areas_json = json.dumps(weak_areas)
        strong_areas_json = json.dumps(strong_areas)
        detailed_analysis_json = json.dumps(detailed_analysis)
        
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO mcq_performance
            (session_hash, total_questions, correct_answers, score_percentage, weak_areas, strong_areas, detailed_analysis)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_hash, total_questions, correct_answers, score_percentage, weak_areas_json, strong_areas_json, detailed_analysis_json))
        conn.commit()
        conn.close()
        print(f"✅ MCQ performance saved for session {session_hash}")
        return True
    except Exception as e:
        print(f"❌ Error saving MCQ performance: {e}")
        return False

def get_user_settings():
    """Get user settings from database."""
    default_settings = {
        'theme': 'dark', 
        'ai_model': 'gemini-2.5-flash',
        'voice_enabled': True,
        'voice_speed': 1.0,
        'voice_gender': 'female',
        'voice_language': 'en',
        'mcq_count': 35
    }
    
    try:
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        c.execute('SELECT theme, ai_model, voice_enabled, voice_speed, voice_gender, voice_language, mcq_count FROM user_settings ORDER BY id DESC LIMIT 1')
        result = c.fetchone()
        conn.close()
        
        if result:
            return {
                'theme': result[0], 
                'ai_model': result[1],
                'voice_enabled': bool(result[2]),
                'voice_speed': result[3],
                'voice_gender': result[4],
                'voice_language': result[5],
                'mcq_count': result[6] if result[6] is not None else 35
            }
        
        return default_settings
        
    except Exception as e:
        print(f"❌ Error getting user settings: {e}")
        return default_settings

def save_user_settings(theme, ai_model, voice_enabled, voice_speed, voice_gender, voice_language, mcq_count):
    """Save user settings to database."""
    try:
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        # First delete any existing settings to ensure only one row exists
        c.execute('DELETE FROM user_settings')
        c.execute('''
            INSERT INTO user_settings (theme, ai_model, voice_enabled, voice_speed, voice_gender, voice_language, mcq_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (theme, ai_model, int(voice_enabled), voice_speed, voice_gender, voice_language, mcq_count))
        conn.commit()
        conn.close()
        print(f"✅ User settings saved")
        return True
    except Exception as e:
        print(f"❌ Error saving user settings: {e}")
        return False

def save_chat_session(filename, formulas, chat_history):
    """Save chat session to database."""
    try:
        session_hash = hashlib.md5(f"{filename}{datetime.now()}".encode()).hexdigest()[:10]
        formulas_json = json.dumps(formulas)
        chat_history_json = json.dumps(chat_history)
        
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO chat_sessions
            (session_hash, filename, extracted_formulas, chat_history)
            VALUES (?, ?, ?, ?)
        ''', (session_hash, filename, formulas_json, chat_history_json))
        conn.commit()
        conn.close()
        print(f"✅ Chat session for {filename} saved successfully.")
        return session_hash
    except Exception as e:
        print(f"❌ Error saving chat session: {e}")
        return None

def get_study_history():
    """Retrieve study session history."""
    try:
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        c.execute('SELECT id, session_hash, filename, content_preview, created_at, key_concepts_count, flashcards_count, mcqs_count FROM study_sessions ORDER BY created_at DESC LIMIT 10')
        sessions = c.fetchall()
        conn.close()
        
        return [{
            'id': session[0],
            'session_hash': session[1],
            'filename': session[2],
            'content_preview': session[3],
            'created_at': session[4],
            'key_concepts_count': session[5],
            'flashcards_count': session[6],
            'mcqs_count': session[7]
        } for session in sessions]
    except Exception as e:
        print(f"❌ Error retrieving study history: {e}")
        return []

def get_study_session_by_hash(session_hash):
    """Retrieve a specific study session by its hash."""
    try:
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        c.execute('SELECT full_content FROM study_sessions WHERE session_hash = ?', (session_hash,))
        result = c.fetchone()
        conn.close()
        if result:
            return json.loads(result[0])
        return None
    except Exception as e:
        print(f"❌ Error retrieving study session: {e}")
        return None

def text_to_speech(text, language='en', speed=1.0, gender='female'):
    """Convert text to speech and return base64 encoded audio."""
    try:
        is_slow = speed < 0.75 
        
        tts = gTTS(text=text, lang=language, slow=is_slow)
        
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        audio_data = mp3_fp.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        return audio_base64
        
    except Exception as e:
        print(f"❌ Error in text-to-speech: {e}")
        return None

def stop_audio():
    """Stop currently playing audio."""
    global current_audio, audio_playing
    try:
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        audio_playing = False
        current_audio = None
        return True
    except Exception as e:
        print(f"❌ Error stopping audio: {e}")
        return False

def play_audio_base64(audio_base64):
    """Play audio using Pygame (for server-side testing)."""
    def play_audio():
        global current_audio, audio_playing
        try:
            audio_data = base64.b64decode(audio_base64)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            pygame.mixer.music.load(tmp_file_path)
            pygame.mixer.music.play()
            audio_playing = True
            current_audio = tmp_file_path
            
            while pygame.mixer.music.get_busy() and audio_playing:
                time.sleep(0.1)
                
            if current_audio == tmp_file_path:
                try:
                    os.unlink(tmp_file_path)
                    current_audio = None
                except:
                    pass
                
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
    
    thread = threading.Thread(target=play_audio)
    thread.daemon = True
    thread.start()

def generate_all_content(notes_text, mcq_count=35):
    """Generates a comprehensive study pack using the Gemini API."""
    if not model:
        return {"error": "Generative AI model is not configured. Please check your API key."}
    
    if not notes_text.strip():
        return {"error": "No text content found in the uploaded file."}
    
    print(f"📝 Generating content from {len(notes_text)} characters of text")
    notes_text = notes_text[:10000]
    
    # Fixed MCQ count between 30-40
    mcq_count = max(30, min(40, mcq_count))
    
    # Calculate distribution based on requested MCQ count
    easy_count = max(1, int(mcq_count * 0.3))
    medium_count = max(1, int(mcq_count * 0.4))
    hard_count = max(1, mcq_count - easy_count - medium_count)
    
    prompt = f"""
    Analyze the following study notes and generate a comprehensive, detailed study pack. 
    Return ONLY valid JSON with this exact structure:

    {{
        "key_concepts": [
            {{
                "concept": "Concept name",
                "explanation": "Detailed 2-3 sentence explanation",
                "importance": "high/medium/low"
            }}
        ],
        "flashcards": [
            {{
                "question": "Specific and challenging question",
                "answer": "Comprehensive answer with details", 
                "category": "Relevant category"
            }}
        ],
        "mcqs": [
            {{
                "question": "Thought-provoking multiple choice question",
                "options": ["Plausible option A", "Plausible option B", "Plausible option C", "Plausible option D"],
                "answer_text": "Detailed correct answer explanation",
                "answer_letter": "A/B/C/D",
                "explanation": "Comprehensive explanation of why the answer is correct and others are wrong",
                "difficulty": "easy/medium/hard",
                "category": "Specific topic category"
            }}
        ],
        "mind_map": {{
            "central_topic": "Main topic from the notes",
            "main_branches": [
                {{
                    "name": "Major branch topic",
                    "sub_branches": ["Detailed sub-topic 1", "Detailed sub-topic 2", "Detailed sub-topic 3", "Detailed sub-topic 4"]
                }}
            ],
            "connections": "Detailed explanation of relationships and connections between concepts"
        }},
        "memory_tricks": {{
            "acronyms": "Multiple creative acronyms and mnemonics",
            "rhymes": "Memorable rhymes and songs with details",
            "visual_associations": "Vivid visual imagery and associations", 
            "story_method": "Engaging story that connects multiple concepts"
        }}
    }}

    QUANTITY REQUIREMENTS:
    - key_concepts: Generate 15-20 comprehensive key concepts
    - flashcards: Generate 12-15 detailed Q&A flashcards  
    - mcqs: Generate EXACTLY {mcq_count} challenging multiple choice questions covering ALL topics and formulas
    - mind_map: Include 4-6 main branches, each with 3-5 detailed sub-branches
    - memory_tricks: Provide extensive, creative memory aids for each technique

    MCQ REQUIREMENTS:
    - Generate EXACTLY {mcq_count} questions total (between 30-40)
    - Difficulty distribution: {easy_count} easy, {medium_count} medium, {hard_count} hard questions
    - Cover ALL topics and formulas found in the document
    - Each question should test different aspects of understanding
    - Include formula-based questions where applicable
    - Ensure diverse question types and categories

    CONTENT QUALITY:
    - Make explanations thorough and educational
    - Ensure questions are challenging and thought-provoking
    - Create detailed, interconnected mind maps
    - Develop creative, memorable memory techniques

    Study Notes to analyze:
    {notes_text}

    Return ONLY the JSON object, no additional text or explanations.
    """

    try:
        print("🔄 Calling Gemini API for comprehensive content...")
        response = model.generate_content(prompt)
        print("✅ API response received")
        
        text = response.text.strip()
        print(f"📋 Raw response length: {len(text)} characters")
        
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
        
        print("🔄 Parsing JSON...")
        result = json.loads(text)
        print("✅ JSON parsed successfully")
        
        required_keys = ["key_concepts", "flashcards", "mcqs", "mind_map", "memory_tricks"]
        for key in required_keys:
            if key not in result:
                if key in ["key_concepts", "flashcards", "mcqs"]:
                    result[key] = []
                else:
                    result[key] = {}
        
        print(f"📊 Generated content stats:")
        print(f"   - Key Concepts: {len(result.get('key_concepts', []))}")
        print(f"   - Flashcards: {len(result.get('flashcards', []))}")
        print(f"   - MCQs: {len(result.get('mcqs', []))}")
        
        if 'mind_map' in result and result['mind_map']:
            mind_map = result['mind_map']
            if 'main_branches' not in mind_map:
                mind_map['main_branches'] = []
            if 'central_topic' not in mind_map:
                mind_map['central_topic'] = "Main Topic"
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"📋 Problematic response preview: {text[:500]}...")
        
        return {
            "key_concepts": [
                {
                    "concept": "AI Content Generation Failed", 
                    "explanation": "The AI response could not be parsed as valid JSON. Please try again or check your API key. Showing fallback content.",
                    "importance": "high"
                }
            ] * 5,
            "flashcards": [
                {
                    "question": "What is the primary cause of this error?",
                    "answer": "The AI returned improperly formatted JSON. Please try running the generation again.",
                    "category": "Troubleshooting"
                }
            ] * 8,
            "mcqs": [
                {
                    "question": "Why might content generation fail?",
                    "options": [
                        "API key issues", 
                        "Network connectivity problems", 
                        "AI formatting error", 
                        "All of the above"
                    ],
                    "answer_text": "All of the above can cause content generation to fail",
                    "answer_letter": "D",
                    "explanation": "API key issues, network problems, and AI formatting errors can all contribute to generation failures.",
                    "difficulty": "easy",
                    "category": "Troubleshooting"
                }
            ] * min(mcq_count, 15),
            "mind_map": {
                "central_topic": "Study Success Strategies",
                "main_branches": [
                    {
                        "name": "Preparation",
                        "sub_branches": ["Check API Key", "Verify Internet", "Optimize File Size", "Review Documentation"]
                    }
                ],
                "connections": "Proper preparation leads to successful execution, while good troubleshooting skills ensure continuous learning improvement."
            },
            "memory_tricks": {
                "acronyms": "API - Always Prepare Intelligently",
                "rhymes": "When content fails to generate, don't hesitate - check your key, and you'll see, learning will flow naturally",
                "visual_associations": "Imagine a key opening a treasure chest of knowledge", 
                "story_method": "A student prepares for exams by first ensuring their tools work properly"
            },
            "error": f"JSONDecodeError: {str(e)} - The AI response format was invalid."
        }
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return {"error": f"AI generation failed: {str(e)}"}

def chat_with_ai(question, context_text, formulas, chat_history):
    """Chat with AI about the study material or general questions."""
    if not model:
        return "AI model is not configured. Please check your API key."
    
    # FAQ and general questions
    faqs = {
        "how to use": "Upload a PDF file using the upload area, then click 'Generate Study Pack' to create flashcards, MCQs, and other study materials. Use the chatbot for questions about your document.",
        "what can you do": "I can generate study materials from PDFs, create flashcards and MCQs, explain concepts, help with formulas, and track your study progress.",
        "features": "Key features: PDF-to-study-pack conversion, interactive flashcards, MCQ practice with performance analysis, mind maps, memory techniques, study history, and voice assistance.",
        "supported files": "I support PDF files up to 100MB. The text extraction works best with text-based PDFs (not scanned images).",
        "mcq count": "I generate 30-40 MCQs automatically to provide comprehensive coverage of your study material.",
        "voice settings": "Voice features are enabled by default. You can interact with SRbot using the chat interface.",
        "study history": "Your generated study packs are saved in the Study History section. Click on any session to reload it.",
        "formulas": "I automatically extract mathematical formulas from your documents for easy reference during study sessions.",
        "theme": "You can change the theme using the theme selector in the navigation bar for a personalized experience."
    }
    
    # Check if it's a general question
    question_lower = question.lower()
    for key in faqs:
        if key in question_lower:
            return faqs[key]
    
    # If no document context, provide general assistance
    if not context_text or context_text.strip() == "":
        general_prompt = f"""
        You are SRbot, an AI study assistant for the AI Study Suite Pro application. 
        The user is asking: "{question}"
        
        About the application:
        - Converts PDFs to comprehensive study packs (flashcards, MCQs, mind maps)
        - Includes interactive features and performance tracking
        - Has voice assistance and multiple theme options
        - Supports study history and formula extraction
        - Generates 30-40 MCQs automatically for comprehensive coverage
        
        Provide a helpful response about how they can use the application or answer their general study-related questions.
        Keep responses friendly, informative, and focused on study assistance.
        
        Response:
        """
        
        try:
            response = model.generate_content(general_prompt)
            return response.text.strip()
        except Exception as e:
            return f"I'm here to help with your studies! You can upload PDFs to generate study materials, or ask me about specific study techniques. Error: {str(e)}"
    
    # Document-specific chat
    formulas_text = "\n".join([f"• {formula}" for formula in formulas]) if formulas else "No formulas found in the document."
    
    prompt = f"""
    You are an expert tutor helping a student understand their study material. 
    Use the following context from their uploaded document to provide accurate, helpful answers.
    
    DOCUMENT CONTEXT:
    {context_text[:3000]}
    
    FORMULAS FROM DOCUMENT:
    {formulas_text}
    
    CHAT HISTORY:
    {chat_history[-5:] if chat_history else "No previous conversation"}
    
    CURRENT QUESTION: {question}
    
    Instructions:
    - Answer based strictly on the document context provided when relevant
    - Reference specific formulas when applicable to the question
    - Provide clear, educational explanations
    - If the question can't be answered from the context, say so politely but try to help generally
    - Keep responses concise but informative
    - Use examples from the context when helpful
    
    Response:
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

def generate_daily_revision_test():
    """Generate daily revision test with AI-powered question selection."""
    try:
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        
        # Get all study sessions with their performance data
        c.execute('''
            SELECT s.session_hash, s.filename, s.full_content, 
                   p.score_percentage, p.weak_areas
            FROM study_sessions s
            LEFT JOIN mcq_performance p ON s.session_hash = p.session_hash
            ORDER BY s.created_at DESC
            LIMIT 10
        ''')
        
        sessions = c.fetchall()
        
        if not sessions:
            return None
        
        # Collect all MCQs and prioritize based on performance
        all_mcqs = []
        question_weights = []
        
        for session_hash, filename, full_content, score_percentage, weak_areas_json in sessions:
            try:
                content = json.loads(full_content)
                mcqs = content.get('mcqs', [])
                
                # Parse weak areas to prioritize questions from weak topics
                weak_areas = []
                if weak_areas_json:
                    weak_areas = json.loads(weak_areas_json)
                
                for mcq in mcqs:
                    weight = 1.0
                    
                    # Increase weight for questions from weak areas
                    category = mcq.get('category', '')
                    if any(weak_area in category for weak_area in weak_areas):
                        weight *= 2.0
                    
                    # Increase weight for harder questions if performance is good
                    if score_percentage and score_percentage > 70:
                        if mcq.get('difficulty') == 'hard':
                            weight *= 1.5
                    
                    all_mcqs.append(mcq)
                    question_weights.append(weight)
                    
            except Exception as e:
                print(f"Error processing session {session_hash}: {e}")
                continue
        
        if not all_mcqs:
            return None
        
        # Select questions based on weights
        total_questions = min(25, len(all_mcqs))
        selected_indices = random.choices(
            range(len(all_mcqs)), 
            weights=question_weights, 
            k=total_questions
        )
        
        selected_mcqs = [all_mcqs[i] for i in selected_indices]
        
        # Save to daily revision tests
        today = date.today().isoformat()
        questions_json = json.dumps(selected_mcqs)
        
        c.execute('''
            INSERT OR REPLACE INTO daily_revision_tests 
            (test_date, questions, completed, score, analysis_data)
            VALUES (?, ?, 0, 0, ?)
        ''', (today, questions_json, '{}'))
        
        conn.commit()
        conn.close()
        
        return selected_mcqs
        
    except Exception as e:
        print(f"❌ Error generating daily revision test: {e}")
        return None

def get_daily_revision_test():
    """Get today's revision test."""
    try:
        today = date.today().isoformat()
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        
        c.execute('SELECT questions, completed, score, analysis_data FROM daily_revision_tests WHERE test_date = ?', (today,))
        result = c.fetchone()
        conn.close()
        
        if result:
            questions = json.loads(result[0]) if result[0] else []
            return {
                'questions': questions,
                'completed': bool(result[1]),
                'score': result[2],
                'analysis_data': json.loads(result[3]) if result[3] else {}
            }
        return None
    except Exception as e:
        print(f"❌ Error getting daily revision test: {e}")
        return None

def generate_ai_analysis(score, questions_data):
    """Generate AI-powered analysis using Gemini API."""
    if not model:
        return get_fallback_analysis(score)
    
    try:
        # Prepare data for AI analysis
        total_questions = len(questions_data)
        correct_answers = sum(1 for q in questions_data if q.get('correct', False))
        incorrect_answers = total_questions - correct_answers
        
        # Analyze question categories and difficulties
        categories = {}
        difficulties = {'easy': 0, 'medium': 0, 'hard': 0}
        
        for q in questions_data:
            category = q.get('category', 'General')
            difficulty = q.get('difficulty', 'medium').lower()
            is_correct = q.get('correct', False)
            
            if category not in categories:
                categories[category] = {'total': 0, 'correct': 0}
            
            categories[category]['total'] += 1
            if is_correct:
                categories[category]['correct'] += 1
            
            if difficulty in difficulties:
                difficulties[difficulty] += 1
        
        # Calculate category performance
        weak_categories = []
        strong_categories = []
        
        for category, data in categories.items():
            accuracy = (data['correct'] / data['total']) * 100
            if accuracy < 60:
                weak_categories.append(f"{category} ({accuracy:.1f}%)")
            elif accuracy >= 85:
                strong_categories.append(f"{category} ({accuracy:.1f}%)")
        
        prompt = f"""
        Analyze this test performance and provide detailed insights:
        
        SCORE: {score}%
        TOTAL QUESTIONS: {total_questions}
        CORRECT ANSWERS: {correct_answers}
        INCORRECT ANSWERS: {incorrect_answers}
        
        PERFORMANCE BREAKDOWN:
        - Weak Categories: {', '.join(weak_categories) if weak_categories else 'None identified'}
        - Strong Categories: {', '.join(strong_categories) if strong_categories else 'None identified'}
        - Difficulty Distribution: Easy: {difficulties['easy']}, Medium: {difficulties['medium']}, Hard: {difficulties['hard']}
        
        Please provide a comprehensive analysis with:
        1. Overall performance assessment
        2. Specific knowledge gaps identified
        3. Memory retention issues spotted
        4. Personalized improvement recommendations
        5. Study strategy suggestions
        
        Format the response as JSON with these keys:
        - overall_assessment
        - knowledge_gaps
        - memory_decay_areas
        - improvement_recommendations
        - study_strategy
        
        Be specific, actionable, and educational.
        """
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean JSON response
        if text.startswith('```json'):
            text = text[7:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
        
        analysis = json.loads(text)
        return analysis
        
    except Exception as e:
        print(f"❌ Error generating AI analysis: {e}")
        return get_fallback_analysis(score)

def get_fallback_analysis(score):
    """Provide fallback analysis when AI is unavailable."""
    if score >= 90:
        return {
            "overall_assessment": "Excellent performance! You have strong mastery of the material.",
            "knowledge_gaps": ["Minor conceptual refinements needed", "Advanced application practice"],
            "memory_decay_areas": ["Long-term retention maintenance", "Complex formula recall"],
            "improvement_recommendations": [
                "Focus on advanced applications",
                "Practice speed and accuracy",
                "Review complex scenarios"
            ],
            "study_strategy": "Maintain current study habits with occasional challenging exercises"
        }
    elif score >= 70:
        return {
            "overall_assessment": "Good performance with room for improvement in specific areas.",
            "knowledge_gaps": ["Intermediate concept application", "Problem-solving strategies"],
            "memory_decay_areas": ["Recent topic retention", "Formula applications"],
            "improvement_recommendations": [
                "Focus on weak categories identified",
                "Practice application questions",
                "Review foundational concepts"
            ],
            "study_strategy": "Balanced review of strong and weak areas with increased practice"
        }
    else:
        return {
            "overall_assessment": "Needs significant improvement. Focus on foundational concepts.",
            "knowledge_gaps": ["Basic concept understanding", "Fundamental principles"],
            "memory_decay_areas": ["Core concept retention", "Basic formula recall"],
            "improvement_recommendations": [
                "Start with basic concepts",
                "Use flashcards for memorization",
                "Practice simple applications first"
            ],
            "study_strategy": "Structured learning starting from fundamentals with regular review"
        }

def save_daily_revision_result(score, analysis_data=None):
    """Save daily test results with AI-generated analysis."""
    try:
        today = date.today().isoformat()
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        
        # Get the test questions to generate AI analysis
        c.execute('SELECT questions FROM daily_revision_tests WHERE test_date = ?', (today,))
        result = c.fetchone()
        
        if result and result[0]:
            questions = json.loads(result[0])
            # Generate AI analysis
            ai_analysis = generate_ai_analysis(score, questions)
            
            analysis_data = analysis_data or {}
            analysis_data['ai_analysis'] = ai_analysis
            
            analysis_json = json.dumps(analysis_data)
            
            c.execute('''
                UPDATE daily_revision_tests 
                SET completed = 1, score = ?, analysis_data = ?
                WHERE test_date = ?
            ''', (score, analysis_json, today))
            
            conn.commit()
            conn.close()
            print(f"✅ Daily test result saved: {score}%")
            return True
        
        conn.close()
        return False
        
    except Exception as e:
        print(f"❌ Error saving daily test result: {e}")
        return False

def save_test_score(test_date, score, test_type, duration):
    """Save test score for timer functionality."""
    try:
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO test_scores (test_date, score, test_type, duration)
            VALUES (?, ?, ?, ?)
        ''', (test_date, score, test_type, duration))
        
        conn.commit()
        conn.close()
        print(f"✅ Test score saved: {score}% for {test_type}")
        return True
    except Exception as e:
        print(f"❌ Error saving test score: {e}")
        return False

def get_test_scores(days=30):
    """Get test scores for the last N days."""
    try:
        start_date = (date.today() - timedelta(days=days)).isoformat()
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        
        c.execute('''
            SELECT test_date, score, test_type, duration 
            FROM test_scores 
            WHERE test_date >= ? 
            ORDER BY test_date ASC
        ''', (start_date,))
        
        results = c.fetchall()
        conn.close()
        
        scores = []
        for result in results:
            scores.append({
                'test_date': result[0],
                'score': result[1],
                'test_type': result[2],
                'duration': result[3]
            })
        
        print(f"✅ Retrieved {len(scores)} test scores")
        return scores
    except Exception as e:
        print(f"❌ Error getting test scores: {e}")
        return []

# Timer functionality
def start_daily_test():
    """Start a daily test with timer."""
    try:
        # Record test start time
        test_start_time = datetime.now()
        end_time = test_start_time + timedelta(minutes=DEFAULT_TEST_DURATION)
        
        test_session = {
            'start_time': test_start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration': DEFAULT_TEST_DURATION,
            'questions_answered': 0,
            'correct_answers': 0
        }
        
        # Store in session
        session['test_session'] = test_session
        
        return jsonify({
            'success': True,
            'test_session': test_session,
            'remaining_time': DEFAULT_TEST_DURATION * 60  # in seconds
        })
        
    except Exception as e:
        print(f"❌ Error starting daily test: {e}")
        return jsonify({'error': str(e)}), 500

def check_test_time():
    """Check remaining test time."""
    try:
        if 'test_session' not in session:
            return None
            
        test_session = session['test_session']
        start_time = datetime.fromisoformat(test_session['start_time'])
        now = datetime.now()
        elapsed = (now - start_time).total_seconds()
        remaining = max(0, DEFAULT_TEST_DURATION * 60 - elapsed)
        
        return {
            'elapsed_time': elapsed,
            'remaining_time': remaining,
            'is_time_up': remaining <= 0
        }
        
    except Exception as e:
        print(f"❌ Error checking test time: {e}")
        return None

def end_daily_test(score, analysis_data=None):
    """End daily test and record performance."""
    try:
        if 'test_session' not in session:
            return jsonify({'error': 'No active test session found'}), 400
            
        test_session = session['test_session']
        start_time = datetime.fromisoformat(test_session['start_time'])
        end_time = datetime.now()
        duration = int((end_time - start_time).total_seconds() / 60)
        
        # Save test result with duration
        test_date = date.today().isoformat()
        save_test_score(test_date, score, 'daily_test', duration)
        
        # Save daily test result
        save_daily_revision_result(score, analysis_data)
        
        # Clear test session
        session.pop('test_session', None)
        
        return jsonify({
            'success': True,
            'test_duration': duration,
            'score': score
        })
        
    except Exception as e:
        print(f"❌ Error ending daily test: {e}")
        return jsonify({'error': str(e)}), 500

# Goals/To-do list functionality
def get_user_goals():
    """Get user goals from database."""
    try:
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        c.execute('SELECT id, title, description, category, priority, due_date, completed FROM user_goals ORDER BY completed ASC, priority DESC, due_date ASC')
        goals = c.fetchall()
        conn.close()
        
        return [{
            'id': goal[0],
            'title': goal[1],
            'description': goal[2],
            'category': goal[3],
            'priority': goal[4],
            'due_date': goal[5],
            'completed': bool(goal[6])
        } for goal in goals]
    except Exception as e:
        print(f"❌ Error getting user goals: {e}")
        return []

def save_user_goal(title, description, category, priority, due_date):
    """Save user goal to database."""
    try:
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO user_goals (title, description, category, priority, due_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (title, description, category, priority, due_date))
        conn.commit()
        conn.close()
        print(f"✅ Goal saved: {title}")
        return True
    except Exception as e:
        print(f"❌ Error saving goal: {e}")
        return False

def update_goal_status(goal_id, completed):
    """Update goal completion status."""
    try:
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        c.execute('UPDATE user_goals SET completed = ? WHERE id = ?', (int(completed), goal_id))
        conn.commit()
        conn.close()
        print(f"✅ Goal {goal_id} updated to completed: {completed}")
        return True
    except Exception as e:
        print(f"❌ Error updating goal: {e}")
        return False

def delete_goal(goal_id):
    """Delete goal from database."""
    try:
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        c.execute('DELETE FROM user_goals WHERE id = ?', (goal_id,))
        conn.commit()
        conn.close()
        print(f"✅ Goal {goal_id} deleted")
        return True
    except Exception as e:
        print(f"❌ Error deleting goal: {e}")
        return False

# Focus mode functionality
def save_focus_session(duration, session_type, notes):
    """Save focus session to database."""
    try:
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO focus_sessions (duration, session_type, notes)
            VALUES (?, ?, ?)
        ''', (duration, session_type, notes))
        conn.commit()
        conn.close()
        print(f"✅ Focus session saved: {duration} minutes")
        return True
    except Exception as e:
        print(f"❌ Error saving focus session: {e}")
        return False

def get_focus_sessions(days=30):
    """Get focus sessions for the last N days."""
    try:
        start_date = (date.today() - timedelta(days=days)).isoformat()
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        c.execute('''
            SELECT duration, session_type, notes, created_at 
            FROM focus_sessions 
            WHERE date(created_at) >= ? 
            ORDER BY created_at DESC
        ''', (start_date,))
        
        results = c.fetchall()
        conn.close()
        
        sessions = []
        for result in results:
            sessions.append({
                'duration': result[0],
                'session_type': result[1],
                'notes': result[2],
                'created_at': result[3]
            })
        
        return sessions
    except Exception as e:
        print(f"❌ Error getting focus sessions: {e}")
        return []

# Enhanced Test Session Management
def save_test_session(session_hash, test_type, questions, user_answers, score, time_taken):
    """Save test session to database."""
    try:
        questions_json = json.dumps(questions)
        user_answers_json = json.dumps(user_answers)
        
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO test_sessions 
            (session_hash, test_type, questions, user_answers, score, time_taken, completed)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        ''', (session_hash, test_type, questions_json, user_answers_json, score, time_taken))
        conn.commit()
        conn.close()
        print(f"✅ Test session saved: {test_type} - Score: {score}%")
        return True
    except Exception as e:
        print(f"❌ Error saving test session: {e}")
        return False

def get_test_sessions(days=30):
    """Get test sessions for the last N days."""
    try:
        start_date = (date.today() - timedelta(days=days)).isoformat()
        conn = sqlite3.connect('study_history.db')
        c = conn.cursor()
        c.execute('''
            SELECT session_hash, test_type, score, time_taken, created_at 
            FROM test_sessions 
            WHERE date(created_at) >= ? 
            ORDER BY created_at DESC
        ''', (start_date,))
        
        results = c.fetchall()
        conn.close()
        
        sessions = []
        for result in results:
            sessions.append({
                'session_hash': result[0],
                'test_type': result[1],
                'score': result[2],
                'time_taken': result[3],
                'created_at': result[4]
            })
        
        return sessions
    except Exception as e:
        print(f"❌ Error getting test sessions: {e}")
        return []

# Routes
@app.route('/')
def index():
    user_settings = get_user_settings()
    study_history = get_study_history()
    daily_test = get_daily_revision_test()
    user_goals = get_user_goals()
    focus_sessions = get_focus_sessions(7)
    test_sessions = get_test_sessions(30)
    
    # Calculate goals statistics
    total_goals = len(user_goals)
    completed_goals = sum(1 for goal in user_goals if goal['completed'])
    pending_goals = total_goals - completed_goals
    
    # Calculate focus session statistics
    total_focus_time = sum(session['duration'] for session in focus_sessions)
    avg_focus_session = total_focus_time / len(focus_sessions) if focus_sessions else 0
    
    # Get test scores for analytics
    test_scores = get_test_scores(30)
    
    # Prepare HTML content with all the premium effects
    html_content = f"""
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Study Suite Pro</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root, body.theme-light {{
            --primary: #3b82f6;
            --secondary: #1e40af;
            --accent: #8b5cf6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --info: #06b6d4;
            --muted: #6b7280;
            --card-bg: #ffffff;
            --text: #111827;
            --border: rgba(0,0,0,0.08);
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.04);
            --shadow-md: 0 8px 25px rgba(0,0,0,0.08);
            --shadow-lg: 0 15px 40px rgba(0,0,0,0.12);
            --shadow-xl: 0 25px 50px rgba(0,0,0,0.15);
            --shadow-hover: 0 25px 60px rgba(0,0,0,0.2);
            --glass: rgba(255,255,255,0.85);
            --glass-dark: rgba(255,255,255,0.95);
            --radius: 16px;
            --radius-sm: 8px;
            --radius-lg: 20px;
            --gradient-main: linear-gradient(135deg, var(--primary), var(--accent));
            --gradient-main-hover: linear-gradient(135deg, var(--accent), var(--primary));
            --gradient-glass: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
            --gradient-dark: linear-gradient(135deg, rgba(0,0,0,0.1), rgba(0,0,0,0.05));
            --sidebar-width: 280px;
            --sidebar-collapsed: 80px;
            --header-height: 70px;
            --transition-slow: 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            --transition-normal: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-fast: 0.15s ease;
            --blur-sm: blur(8px);
            --blur-md: blur(16px);
            --blur-lg: blur(24px);
        }}

        body.theme-dark {{
            --primary: #60a5fa;
            --secondary: #3b82f6;
            --accent: #a78bfa;
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
            --info: #22d3ee;
            --muted: #9ca3af;
            --card-bg: #1f2937;
            --text: #f9fafb;
            --border: rgba(255,255,255,0.1);
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
            --shadow-md: 0 8px 25px rgba(0,0,0,0.4);
            --shadow-lg: 0 15px 40px rgba(0,0,0,0.5);
            --shadow-xl: 0 25px 50px rgba(0,0,0,0.6);
            --shadow-hover: 0 25px 60px rgba(0,0,0,0.7);
            --glass: rgba(30, 41, 59, 0.8);
            --glass-dark: rgba(30, 41, 59, 0.9);
            --gradient-glass: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
            --gradient-dark: linear-gradient(135deg, rgba(0,0,0,0.2), rgba(0,0,0,0.1));
        }}
        
        body.theme-cosmic {{
            --primary: #8b5cf6;
            --secondary: #7c3aed;
            --accent: #ec4899;
            --success: #34d399;
            --warning: #fb923c;
            --danger: #f87171;
            --info: #22d3ee;
            --muted: #a1a1aa;
            --card-bg: rgba(21, 15, 33, 0.9);
            --text: #e0e7ff;
            --border: rgba(255,255,255,0.15);
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.4);
            --shadow-md: 0 8px 25px rgba(0,0,0,0.5);
            --shadow-lg: 0 15px 40px rgba(0,0,0,0.6);
            --shadow-xl: 0 25px 50px rgba(168, 85, 247, 0.2);
            --shadow-hover: 0 25px 60px rgba(168, 85, 247, 0.3);
            --glass: rgba(21, 15, 33, 0.8);
            --glass-dark: rgba(21, 15, 33, 0.95);
            --gradient-glass: linear-gradient(135deg, rgba(168, 85, 247, 0.1), rgba(236, 72, 153, 0.05));
        }}

        body.theme-sunset {{
            --primary: #f97316;
            --secondary: #ea580c;
            --accent: #f59e0b;
            --success: #34d399;
            --warning: #facc15;
            --danger: #dc2626;
            --info: #0ea5e9;
            --muted: #a3a3a3;
            --card-bg: #ffffff;
            --text: #1f2937;
            --border: rgba(0,0,0,0.1);
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.06);
            --shadow-md: 0 8px 25px rgba(0,0,0,0.1);
            --shadow-lg: 0 15px 40px rgba(0,0,0,0.15);
            --shadow-xl: 0 25px 50px rgba(249, 115, 22, 0.15);
            --shadow-hover: 0 25px 60px rgba(249, 115, 22, 0.2);
            --glass: rgba(255,255,255,0.9);
            --glass-dark: rgba(255,255,255,0.95);
            --gradient-glass: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        }}

        body.theme-ocean {{
            --primary: #06b6d4;
            --secondary: #0891b2;
            --accent: #22d3ee;
            --success: #34d399;
            --warning: #fb923c;
            --danger: #ef4444;
            --info: #0ea5e9;
            --muted: #a1a1aa;
            --card-bg: rgba(12, 74, 110, 0.9);
            --text: #e0f2fe;
            --border: rgba(255,255,255,0.15);
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
            --shadow-md: 0 8px 25px rgba(0,0,0,0.4);
            --shadow-lg: 0 15px 40px rgba(0,0,0,0.5);
            --shadow-xl: 0 25px 50px rgba(6, 182, 212, 0.2);
            --shadow-hover: 0 25px 60px rgba(6, 182, 212, 0.3);
            --glass: rgba(12, 74, 110, 0.8);
            --glass-dark: rgba(12, 74, 110, 0.95);
            --gradient-glass: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(34, 211, 238, 0.05));
        }}

        body.theme-forest {{
            --primary: #15803d;
            --secondary: #166534;
            --accent: #4ade80;
            --success: #34d399;
            --warning: #f59e0b;
            --danger: #ef4444;
            --info: #0ea5e9;
            --muted: #a3a3a3;
            --card-bg: rgba(28, 38, 32, 0.9);
            --text: #dcfce7;
            --border: rgba(255,255,255,0.15);
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
            --shadow-md: 0 8px 25px rgba(0,0,0,0.4);
            --shadow-lg: 0 15px 40px rgba(0,0,0,0.5);
            --shadow-xl: 0 25px 50px rgba(21, 128, 61, 0.2);
            --shadow-hover: 0 25px 60px rgba(21, 128, 61, 0.3);
            --glass: rgba(28, 38, 32, 0.8);
            --glass-dark: rgba(28, 38, 32, 0.95);
            --gradient-glass: linear-gradient(135deg, rgba(21, 128, 61, 0.1), rgba(74, 222, 128, 0.05));
        }}
        
        body.theme-fire {{
            --primary: #f87171;
            --secondary: #ef4444;
            --accent: #fb923c;
            --success: #34d399;
            --warning: #facc15;
            --danger: #dc2626;
            --info: #0ea5e9;
            --muted: #94a3b8;
            --card-bg: rgba(61, 26, 26, 0.9);
            --text: #fee2e2;
            --border: rgba(255,255,255,0.15);
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
            --shadow-md: 0 8px 25px rgba(0,0,0,0.4);
            --shadow-lg: 0 15px 40px rgba(0,0,0,0.5);
            --shadow-xl: 0 25px 50px rgba(248, 113, 113, 0.2);
            --shadow-hover: 0 25px 60px rgba(248, 113, 113, 0.3);
            --glass: rgba(61, 26, 26, 0.8);
            --glass-dark: rgba(61, 26, 26, 0.95);
            --gradient-glass: linear-gradient(135deg, rgba(248, 113, 113, 0.1), rgba(251, 146, 60, 0.05));
        }}
        
        body.theme-midnight {{
            --primary: #5850ec;
            --secondary: #4f46e5;
            --accent: #a78bfa;
            --success: #34d399;
            --warning: #f59e0b;
            --danger: #ef4444;
            --info: #06b6d4;
            --muted: #94a3b8;
            --card-bg: rgba(26, 26, 36, 0.95);
            --text: #f1f5f9;
            --border: rgba(255,255,255,0.1);
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.4);
            --shadow-md: 0 8px 25px rgba(0,0,0,0.5);
            --shadow-lg: 0 15px 40px rgba(0,0,0,0.6);
            --shadow-xl: 0 25px 50px rgba(88, 80, 236, 0.2);
            --shadow-hover: 0 25px 60px rgba(88, 80, 236, 0.3);
            --glass: rgba(26, 26, 36, 0.8);
            --glass-dark: rgba(26, 26, 36, 0.95);
            --gradient-glass: linear-gradient(135deg, rgba(88, 80, 236, 0.1), rgba(167, 139, 250, 0.05));
        }}

        body.theme-synthwave {{
            --primary: #ec4899;
            --secondary: #db2777;
            --accent: #22d3ee;
            --success: #34d399;
            --warning: #facc15;
            --danger: #ef4444;
            --info: #0ea5e9;
            --muted: #a1a1aa;
            --card-bg: rgba(30, 15, 60, 0.9);
            --text: #f0f0ff;
            --border: rgba(255,255,255,0.15);
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.4);
            --shadow-md: 0 8px 25px rgba(0,0,0,0.5);
            --shadow-lg: 0 15px 40px rgba(0,0,0,0.6);
            --shadow-xl: 0 25px 50px rgba(236, 72, 153, 0.2);
            --shadow-hover: 0 25px 60px rgba(236, 72, 153, 0.3);
            --glass: rgba(30, 15, 60, 0.8);
            --glass-dark: rgba(30, 15, 60, 0.95);
            --gradient-glass: linear-gradient(135deg, rgba(236, 72, 153, 0.1), rgba(34, 211, 238, 0.05));
        }}

        body.theme-emerald {{
            --primary: #10b981;
            --secondary: #059669;
            --accent: #34d399;
            --success: #84cc16;
            --warning: #f59e0b;
            --danger: #ef4444;
            --info: #0ea5e9;
            --muted: #a3a3a3;
            --card-bg: rgba(22, 28, 22, 0.95);
            --text: #dcfce7;
            --border: rgba(255,255,255,0.15);
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
            --shadow-md: 0 8px 25px rgba(0,0,0,0.4);
            --shadow-lg: 0 15px 40px rgba(0,0,0,0.5);
            --shadow-xl: 0 25px 50px rgba(16, 185, 129, 0.2);
            --shadow-hover: 0 25px 60px rgba(16, 185, 129, 0.3);
            --glass: rgba(22, 28, 22, 0.8);
            --glass-dark: rgba(22, 28, 22, 0.95);
            --gradient-glass: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(52, 211, 153, 0.05));
        }}

        body.theme-volcano {{
            --primary: #f97316;
            --secondary: #ea580c;
            --accent: #ef4444;
            --success: #34d399;
            --warning: #facc15;
            --danger: #dc2626;
            --info: #0ea5e9;
            --muted: #94a3b8;
            --card-bg: rgba(42, 42, 42, 0.95);
            --text: #f1f5f9;
            --border: rgba(255,255,255,0.1);
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
            --shadow-md: 0 8px 25px rgba(0,0,0,0.4);
            --shadow-lg: 0 15px 40px rgba(0,0,0,0.5);
            --shadow-xl: 0 25px 50px rgba(249, 115, 22, 0.2);
            --shadow-hover: 0 25px 60px rgba(249, 115, 22, 0.3);
            --glass: rgba(42, 42, 42, 0.8);
            --glass-dark: rgba(42, 42, 42, 0.95);
            --gradient-glass: linear-gradient(135deg, rgba(249, 115, 22, 0.1), rgba(239, 68, 68, 0.05));
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, var(--card-bg) 0%, var(--glass) 100%);
            color: var(--text);
            min-height: 100vh;
            overflow-x: hidden;
            transition: all var(--transition-normal);
            position: relative;
        }}

        body::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(6, 182, 212, 0.05) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }}

        .app-container {{
            display: flex;
            min-height: 100vh;
            position: relative;
        }}

        /* Premium Enhanced Sidebar Navigation */
        .sidebar {{
            width: var(--sidebar-width);
            background: var(--glass);
            backdrop-filter: var(--blur-lg);
            border-right: 1px solid var(--border);
            transition: all var(--transition-slow);
            position: fixed;
            height: 100vh;
            z-index: 1000;
            overflow-y: auto;
            overflow-x: hidden;
            box-shadow: var(--shadow-xl);
            transform: translateX(0);
        }}

        .sidebar.collapsed {{
            width: var(--sidebar-collapsed);
            transform: translateX(0);
        }}

        .sidebar.collapsed .sidebar-content {{
            opacity: 0;
            transform: translateX(-20px);
            pointer-events: none;
        }}

        .sidebar-header {{
            padding: 1.5rem 1rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 1rem;
            position: relative;
            background: var(--glass-dark);
            backdrop-filter: var(--blur-md);
        }}

        .sidebar.collapsed .sidebar-header {{
            justify-content: center;
            padding: 1.5rem 0.5rem;
        }}

        .logo {{
            display: flex;
            align-items: center;
            gap: 1rem;
            font-size: 1.2rem;
            font-weight: 800;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            transition: all var(--transition-normal);
        }}

        .logo:hover {{
            transform: scale(1.05);
        }}

        .sidebar.collapsed .logo span {{
            display: none;
        }}

        .toggle-sidebar {{
            background: var(--gradient-glass);
            border: 1px solid var(--border);
            color: var(--text);
            font-size: 1.1rem;
            cursor: pointer;
            padding: 0.6rem;
            border-radius: var(--radius-sm);
            transition: all var(--transition-normal);
            backdrop-filter: var(--blur-sm);
            position: absolute;
            right: 1rem;
            top: 50%;
            transform: translateY(-50%);
        }}

        .toggle-sidebar:hover {{
            background: var(--gradient-main);
            color: white;
            transform: translateY(-50%) scale(1.1);
            box-shadow: var(--shadow-lg);
        }}

        .sidebar.collapsed .toggle-sidebar {{
            position: static;
            transform: none;
            margin: 0 auto;
        }}

        .sidebar.collapsed .toggle-sidebar:hover {{
            transform: scale(1.1);
        }}

        .sidebar-content {{
            padding: 1rem;
            transition: all var(--transition-slow);
            opacity: 1;
            transform: translateX(0);
        }}

        .nav-section {{
            padding: 1rem 0;
            border-bottom: 1px solid var(--border);
        }}

        .nav-section:last-child {{
            border-bottom: none;
        }}

        .nav-title {{
            padding: 0.5rem 0;
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
            transition: all var(--transition-normal);
        }}

        .sidebar.collapsed .nav-title {{
            opacity: 0;
            height: 0;
            margin: 0;
            padding: 0;
        }}

        .nav-items {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .nav-item {{
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.8rem 1rem;
            color: var(--text);
            text-decoration: none;
            transition: all var(--transition-normal);
            border-radius: var(--radius-sm);
            border: 1px solid transparent;
            background: var(--gradient-glass);
            backdrop-filter: var(--blur-sm);
            position: relative;
            overflow: hidden;
            cursor: pointer;
        }}

        .nav-item::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: var(--gradient-main);
            transition: left var(--transition-normal);
            z-index: -1;
        }}

        .nav-item:hover {{
            transform: translateX(5px);
            border-color: var(--primary);
            box-shadow: var(--shadow-md);
        }}

        .nav-item:hover::before {{
            left: 0;
        }}

        .nav-item:hover .nav-icon {{
            transform: scale(1.2);
            color: white;
        }}

        .nav-item:hover .nav-text {{
            color: white;
            transform: translateX(3px);
        }}

        .nav-item.active {{
            background: var(--gradient-main);
            border-color: var(--primary);
            box-shadow: var(--shadow-lg);
            transform: translateX(5px);
        }}

        .nav-item.active::before {{
            left: 0;
        }}

        .nav-item.active .nav-icon,
        .nav-item.active .nav-text {{
            color: white;
        }}

        .nav-icon {{
            font-size: 1.1rem;
            width: 20px;
            text-align: center;
            transition: all var(--transition-normal);
            z-index: 1;
        }}

        .nav-text {{
            font-weight: 600;
            transition: all var(--transition-normal);
            z-index: 1;
            font-size: 0.9rem;
        }}

        .sidebar.collapsed .nav-text {{
            opacity: 0;
            width: 0;
            overflow: hidden;
        }}

        .theme-selector-container {{
            margin-top: 1.5rem;
            padding: 1rem;
            background: var(--glass-dark);
            border-radius: var(--radius);
            border: 1px solid var(--border);
            backdrop-filter: var(--blur-md);
        }}

        .theme-selector-label {{
            display: block;
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.8rem;
        }}

        .theme-selector {{
            width: 100%;
            background: var(--gradient-glass);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 0.8rem;
            color: var(--text);
            cursor: pointer;
            font-weight: 600;
            transition: all var(--transition-normal);
            backdrop-filter: var(--blur-sm);
            position: relative;
            overflow: hidden;
        }}

        .theme-selector::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: var(--gradient-main);
            transition: left var(--transition-normal);
            z-index: -1;
        }}

        .theme-selector:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
            border-color: var(--primary);
        }}

        .theme-selector:hover::before {{
            left: 0;
        }}

        .theme-selector:hover {{
            color: white;
        }}

        .main-content {{
            flex: 1;
            margin-left: var(--sidebar-width);
            transition: margin-left var(--transition-slow);
            min-height: 100vh;
            position: relative;
        }}

        .main-content.expanded {{
            margin-left: var(--sidebar-collapsed);
        }}

        .top-header {{
            height: var(--header-height);
            background: var(--glass);
            backdrop-filter: var(--blur-lg);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: var(--shadow-sm);
        }}

        .page-title {{
            font-size: 1.5rem;
            font-weight: 800;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            transition: all var(--transition-normal);
        }}

        .header-controls {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .content-area {{
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
        }}

        /* Enhanced Test Interface Styles */
        .test-interface {{
            background: var(--glass);
            border-radius: var(--radius-lg);
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
            backdrop-filter: var(--blur-lg);
            position: relative;
            overflow: hidden;
        }}

        .test-interface::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-glass);
            pointer-events: none;
        }}

        .test-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--border);
        }}

        .test-timer {{
            font-size: 1.3rem;
            font-weight: 700;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 0.6rem 1.2rem;
            border-radius: var(--radius);
            border: 2px solid var(--primary);
        }}

        .test-progress {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}

        .progress-bar {{
            flex: 1;
            height: 6px;
            background: var(--border);
            border-radius: 3px;
            overflow: hidden;
        }}

        .progress-fill {{
            height: 100%;
            background: var(--gradient-main);
            border-radius: 3px;
            transition: width 0.3s ease;
        }}

        .question-navigation {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 2px solid var(--border);
        }}

        .nav-buttons {{
            display: flex;
            gap: 0.8rem;
        }}

        .question-number {{
            font-weight: 700;
            color: var(--primary);
            font-size: 1rem;
        }}

        .test-control-btn {{
            padding: 0.8rem 1.5rem;
            border: none;
            border-radius: var(--radius);
            font-weight: 600;
            cursor: pointer;
            transition: all var(--transition-normal);
            font-size: 0.9rem;
        }}

        .nav-btn {{
            background: var(--gradient-glass);
            color: var(--text);
            border: 1px solid var(--border);
        }}

        .nav-btn:hover {{
            background: var(--gradient-main);
            color: white;
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }}

        .submit-btn {{
            background: var(--success);
            color: white;
        }}

        .submit-btn:hover {{
            background: var(--success);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }}

        .exit-btn {{
            background: var(--danger);
            color: white;
        }}

        .exit-btn:hover {{
            background: var(--danger);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }}

        .question-container {{
            background: var(--glass-dark);
            padding: 2rem;
            border-radius: var(--radius-lg);
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
            backdrop-filter: var(--blur-sm);
        }}

        .question-text {{
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 1.5rem;
            line-height: 1.5;
        }}

        .options-container {{
            display: grid;
            gap: 0.8rem;
        }}

        .option-item {{
            padding: 1rem 1.2rem;
            background: var(--glass);
            border-radius: var(--radius);
            cursor: pointer;
            transition: all var(--transition-normal);
            border: 2px solid transparent;
            font-weight: 500;
            backdrop-filter: var(--blur-sm);
            position: relative;
            overflow: hidden;
        }}

        .option-item::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: var(--gradient-main);
            transition: left var(--transition-normal);
            opacity: 0.1;
        }}

        .option-item:hover {{
            border-color: var(--primary);
            transform: translateX(5px);
            box-shadow: var(--shadow-md);
        }}

        .option-item:hover::before {{
            left: 0;
        }}

        .option-item.selected {{
            border-color: var(--primary);
            background: var(--primary);
            color: white;
            transform: translateX(5px);
            box-shadow: var(--shadow-md);
        }}

        .option-item.correct {{
            border-color: var(--success);
            background: var(--success);
            color: white;
        }}

        .option-item.incorrect {{
            border-color: var(--danger);
            background: var(--danger);
            color: white;
        }}

        /* Enhanced Analysis Dashboard */
        .analysis-dashboard {{
            background: var(--glass);
            border-radius: var(--radius-lg);
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
            backdrop-filter: var(--blur-lg);
        }}

        .score-circle {{
            width: 180px;
            height: 180px;
            margin: 0 auto 2rem;
            position: relative;
        }}

        .circle-bg {{
            fill: none;
            stroke: var(--border);
            stroke-width: 8;
        }}

        .circle-progress {{
            fill: none;
            stroke: var(--success);
            stroke-width: 8;
            stroke-linecap: round;
            transform: rotate(-90deg);
            transform-origin: 50% 50%;
            transition: stroke-dasharray 1s ease;
        }}

        .circle-text {{
            font-size: 2rem;
            font-weight: 900;
            fill: var(--text);
        }}

        .circle-label {{
            font-size: 0.9rem;
            fill: var(--muted);
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }}

        .metric-card {{
            background: var(--glass-dark);
            padding: 1.2rem;
            border-radius: var(--radius);
            text-align: center;
            backdrop-filter: var(--blur-sm);
        }}

        .metric-value {{
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--primary);
        }}

        .metric-label {{
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }}

        .charts-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }}

        .chart-card {{
            background: var(--glass-dark);
            padding: 1.5rem;
            border-radius: var(--radius);
            backdrop-filter: var(--blur-sm);
        }}

        .recommendations {{
            background: var(--glass-dark);
            padding: 1.5rem;
            border-radius: var(--radius);
            margin-top: 1.5rem;
            backdrop-filter: var(--blur-sm);
        }}

        .recommendation-item {{
            padding: 0.8rem;
            margin-bottom: 0.8rem;
            background: var(--glass);
            border-radius: var(--radius-sm);
            border-left: 4px solid var(--info);
        }}

        /* Premium Enhanced Chat Bot Styles */
        .chatbot-container {{
            position: fixed;
            bottom: 1.5rem;
            right: 1.5rem;
            z-index: 1000;
        }}

        .chatbot-toggle {{
            width: 70px;
            height: 70px;
            border-radius: 50%;
            background: var(--gradient-main);
            border: none;
            color: white;
            font-size: 1.8rem;
            cursor: pointer;
            box-shadow: var(--shadow-xl);
            transition: all var(--transition-normal);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            animation: float 3s ease-in-out infinite;
        }}

        @keyframes float {{
            0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
            50% {{ transform: translateY(-8px) rotate(5deg); }}
        }}

        .chatbot-toggle::before {{
            content: '🤖';
            font-weight: bold;
        }}

        .chatbot-toggle::after {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            transition: left 0.6s;
        }}

        .chatbot-toggle:hover::after {{
            left: 100%;
        }}

        .chatbot-toggle:hover {{
            transform: scale(1.1) rotate(10deg);
            box-shadow: var(--shadow-hover);
        }}

        .chatbot-window {{
            position: absolute;
            bottom: 80px;
            right: 0;
            width: 380px;
            height: 500px;
            background: var(--glass);
            backdrop-filter: var(--blur-lg);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-xl);
            display: none;
            flex-direction: column;
            overflow: hidden;
            transform: translateY(20px) scale(0.95);
            opacity: 0;
            transition: all var(--transition-normal);
        }}

        .chatbot-window.active {{
            display: flex;
            transform: translateY(0) scale(1);
            opacity: 1;
        }}

        .chatbot-header {{
            padding: 1.2rem;
            background: var(--gradient-main);
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: var(--shadow-sm);
        }}

        .chatbot-header h3 {{
            margin: 0;
            font-weight: 700;
            font-size: 1.1rem;
        }}

        .formula-toggle {{
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 0.5rem 0.8rem;
            border-radius: var(--radius-sm);
            cursor: pointer;
            transition: all var(--transition-normal);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
            font-weight: 600;
            backdrop-filter: var(--blur-sm);
        }}

        .formula-toggle:hover {{
            background: rgba(255,255,255,0.3);
            transform: translateY(-1px);
        }}

        .formulas-sidebar {{
            position: absolute;
            left: -100%;
            top: 0;
            width: 100%;
            height: 100%;
            background: var(--glass);
            backdrop-filter: var(--blur-lg);
            padding: 1.2rem;
            overflow-y: auto;
            transition: left var(--transition-normal);
            z-index: 10;
        }}

        .formulas-sidebar.active {{
            left: 0;
        }}

        .formulas-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.2rem;
            padding-bottom: 0.8rem;
            border-bottom: 1px solid var(--border);
        }}

        .back-button {{
            background: var(--gradient-main);
            color: white;
            border: none;
            padding: 0.5rem 0.8rem;
            border-radius: var(--radius-sm);
            cursor: pointer;
            font-size: 0.8rem;
            font-weight: 600;
            transition: all var(--transition-normal);
        }}

        .back-button:hover {{
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }}

        .formula-item {{
            padding: 0.8rem;
            margin-bottom: 0.6rem;
            background: var(--glass-dark);
            border-radius: var(--radius-sm);
            border-left: 4px solid var(--primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            box-shadow: var(--shadow-sm);
            transition: all var(--transition-normal);
            backdrop-filter: var(--blur-sm);
        }}

        .formula-item:hover {{
            transform: translateX(3px);
            box-shadow: var(--shadow-md);
        }}

        .chatbot-messages {{
            flex: 1;
            padding: 1.2rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
        }}

        .message {{
            max-width: 85%;
            padding: 0.8rem 1rem;
            border-radius: var(--radius);
            line-height: 1.4;
            position: relative;
            animation: messageSlide 0.3s ease;
            font-size: 0.9rem;
            backdrop-filter: var(--blur-sm);
            transition: all var(--transition-normal);
        }}

        @keyframes messageSlide {{
            from {{
                opacity: 0;
                transform: translateY(8px) scale(0.95);
            }}
            to {{
                opacity: 1;
                transform: translateY(0) scale(1);
            }}
        }}

        .message.user {{
            align-self: flex-end;
            background: var(--gradient-main);
            color: white;
            border-bottom-right-radius: var(--radius-sm);
            box-shadow: var(--shadow-md);
        }}

        .message.bot {{
            align-self: flex-start;
            background: var(--glass-dark);
            color: var(--text);
            border-bottom-left-radius: var(--radius-sm);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
        }}

        .message-actions {{
            display: flex;
            gap: 0.4rem;
            margin-top: 0.6rem;
        }}

        .message-action {{
            background: rgba(255,255,255,0.1);
            border: none;
            color: var(--text);
            padding: 0.3rem 0.6rem;
            border-radius: var(--radius-sm);
            cursor: pointer;
            font-size: 0.75rem;
            transition: all var(--transition-normal);
            backdrop-filter: var(--blur-sm);
        }}

        .message.user .message-action {{
            background: rgba(255,255,255,0.2);
            color: white;
        }}

        .message-action:hover {{
            background: var(--primary);
            color: white;
            transform: translateY(-1px);
        }}

        .chatbot-input {{
            padding: 1.2rem;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 0.6rem;
            background: var(--glass-dark);
            backdrop-filter: var(--blur-sm);
        }}

        .chatbot-input input {{
            flex: 1;
            padding: 0.8rem 1rem;
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            background: var(--glass);
            color: var(--text);
            outline: none;
            font-size: 0.9rem;
            transition: all var(--transition-normal);
            backdrop-filter: var(--blur-sm);
        }}

        .chatbot-input input:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
            transform: translateY(-1px);
        }}

        .chatbot-input button {{
            padding: 0.8rem 1.2rem;
            background: var(--gradient-main);
            border: none;
            border-radius: var(--radius-sm);
            color: white;
            cursor: pointer;
            transition: all var(--transition-normal);
            font-size: 0.9rem;
            font-weight: 600;
        }}

        .chatbot-input button:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }}

        /* Premium Enhanced Main Layout */
        .header {{
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem 1.5rem;
            background: var(--glass);
            backdrop-filter: var(--blur-lg);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }}

        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-glass);
            pointer-events: none;
        }}

        .header h1 {{
            font-size: 2.5rem;
            font-weight: 900;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.8rem;
            text-shadow: 0 4px 8px rgba(0,0,0,0.1);
            position: relative;
        }}

        .header p {{
            color: var(--text);
            font-size: 1.1rem;
            opacity: 0.9;
            line-height: 1.5;
            max-width: 600px;
            margin: 0 auto;
            position: relative;
        }}

        /* Premium Enhanced Upload Section */
        .upload-section {{
            background: var(--glass);
            backdrop-filter: var(--blur-lg);
            border-radius: var(--radius-lg);
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-xl);
            transition: all var(--transition-normal);
            position: relative;
            overflow: hidden;
        }}

        .upload-section::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-glass);
            pointer-events: none;
        }}

        .upload-section:hover {{
            transform: translateY(-5px);
            box-shadow: var(--shadow-hover);
        }}

        .upload-area {{
            border: 3px dashed var(--border);
            border-radius: var(--radius-lg);
            padding: 3rem 2rem;
            text-align: center;
            transition: all var(--transition-normal);
            cursor: pointer;
            margin-bottom: 1.5rem;
            background: var(--gradient-glass);
            backdrop-filter: var(--blur-sm);
            position: relative;
            overflow: hidden;
        }}

        .upload-area::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: var(--gradient-main);
            transition: left var(--transition-normal);
            opacity: 0.1;
        }}

        .upload-area:hover {{
            border-color: var(--primary);
            background: var(--glass-dark);
            transform: scale(1.02);
        }}

        .upload-area:hover::before {{
            left: 0;
        }}

        .upload-area i {{
            font-size: 3rem;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            transition: all var(--transition-normal);
        }}

        .upload-area:hover i {{
            transform: scale(1.1) translateY(-3px);
        }}

        .file-input {{
            display: none;
        }}

        .generate-btn {{
            background: var(--gradient-main);
            color: white;
            border: none;
            padding: 1.2rem 2.5rem;
            border-radius: var(--radius);
            font-weight: 700;
            cursor: pointer;
            transition: all var(--transition-normal);
            box-shadow: var(--shadow-xl);
            width: 100%;
            font-size: 1.1rem;
            position: relative;
            overflow: hidden;
            letter-spacing: 0.5px;
        }}

        .generate-btn::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            transition: left 0.6s;
        }}

        .generate-btn:hover::before {{
            left: 100%;
        }}

        .generate-btn:hover {{
            transform: translateY(-3px) scale(1.02);
            box-shadow: var(--shadow-hover);
            background: var(--gradient-main-hover);
        }}

        .generate-btn:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }}

        /* Premium Enhanced Content Sections */
        .content-tabs {{
            display: flex;
            gap: 0.6rem;
            margin-bottom: 1.5rem;
            background: var(--glass);
            padding: 0.6rem;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            flex-wrap: wrap;
            backdrop-filter: var(--blur-md);
            box-shadow: var(--shadow-md);
        }}

        .tab-btn {{
            flex: 1;
            padding: 1rem 1.5rem;
            border: none;
            background: var(--gradient-glass);
            color: var(--text);
            border-radius: var(--radius-sm);
            cursor: pointer;
            transition: all var(--transition-normal);
            font-weight: 600;
            min-width: 140px;
            position: relative;
            overflow: hidden;
            font-size: 0.95rem;
            backdrop-filter: var(--blur-sm);
        }}

        .tab-btn::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: var(--gradient-main);
            transition: left var(--transition-normal);
            z-index: -1;
        }}

        .tab-btn.active {{
            color: white;
            box-shadow: var(--shadow-lg);
        }}

        .tab-btn.active::before {{
            left: 0;
        }}

        .tab-btn:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }}

        .tab-btn:hover::before {{
            left: 0;
        }}

        .tab-btn:hover {{
            color: white;
        }}

        .content-panel {{
            display: none;
            animation: fadeInUp 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }}

        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px) scale(0.95);
            }}
            to {{
                opacity: 1;
                transform: translateY(0) scale(1);
            }}
        }}

        .content-panel.active {{
            display: block;
        }}

        /* Premium Enhanced MCQ Performance Section */
        .mcq-performance {{
            background: var(--glass);
            backdrop-filter: var(--blur-lg);
            border-radius: var(--radius-lg);
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-xl);
            display: none;
            position: relative;
            overflow: hidden;
        }}

        .mcq-performance::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-glass);
            pointer-events: none;
        }}

        .performance-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            position: relative;
        }}

        .score-display {{
            text-align: center;
            padding: 2rem;
            background: var(--gradient-main);
            border-radius: var(--radius);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }}

        .score-display::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        }}

        .score-value {{
            font-size: 3rem;
            font-weight: 900;
            display: block;
            text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}

        .weak-areas {{
            background: var(--glass-dark);
            padding: 1.5rem;
            border-radius: var(--radius);
            border-left: 4px solid var(--warning);
            margin-bottom: 1rem;
            box-shadow: var(--shadow-md);
            backdrop-filter: var(--blur-sm);
            transition: all var(--transition-normal);
        }}

        .weak-areas:hover {{
            transform: translateX(3px);
            box-shadow: var(--shadow-lg);
        }}

        .weak-areas h4 {{
            color: var(--warning);
            margin-bottom: 0.8rem;
            font-size: 1.1rem;
        }}

        .area-list {{
            list-style: none;
        }}

        .area-list li {{
            padding: 0.6rem 0;
            border-bottom: 1px solid var(--border);
            transition: all var(--transition-normal);
        }}

        .area-list li:hover {{
            transform: translateX(3px);
            color: var(--warning);
        }}

        .area-list li:last-child {{
            border-bottom: none;
        }}

        .detailed-analysis-btn {{
            background: var(--gradient-main);
            color: white;
            border: none;
            padding: 1rem 1.5rem;
            border-radius: var(--radius);
            cursor: pointer;
            transition: all var(--transition-normal);
            font-weight: 600;
            width: 100%;
            margin-top: 1rem;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }}

        .detailed-analysis-btn:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
        }}

        .detailed-analysis {{
            background: var(--glass-dark);
            padding: 2rem;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            margin-top: 1.5rem;
            display: none;
            box-shadow: var(--shadow-md);
            backdrop-filter: var(--blur-sm);
        }}

        .analysis-section {{
            margin-bottom: 1.5rem;
            padding: 1.2rem;
            background: var(--glass);
            border-radius: var(--radius-sm);
            border-left: 3px solid var(--primary);
            transition: all var(--transition-normal);
        }}

        .analysis-section:hover {{
            transform: translateX(3px);
            box-shadow: var(--shadow-md);
        }}

        .analysis-section h4 {{
            color: var(--primary);
            margin-bottom: 0.8rem;
            font-size: 1rem;
        }}

        /* Premium Daily Revision Test Section */
        .daily-test-section {{
            background: var(--glass);
            backdrop-filter: var(--blur-lg);
            border-radius: var(--radius-lg);
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-xl);
            position: relative;
            overflow: hidden;
        }}

        .daily-test-section::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-glass);
            pointer-events: none;
        }}

        .test-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            position: relative;
        }}

        .test-title {{
            font-size: 1.8rem;
            font-weight: 800;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .test-description {{
            color: var(--text);
            margin-bottom: 1.5rem;
            line-height: 1.6;
            font-size: 1rem;
        }}

        .test-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}

        .test-stat {{
            background: var(--glass-dark);
            padding: 1.2rem;
            border-radius: var(--radius);
            text-align: center;
            border: 1px solid var(--border);
            backdrop-filter: var(--blur-sm);
            transition: all var(--transition-normal);
        }}

        .test-stat:hover {{
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
        }}

        .test-stat-value {{
            font-size: 1.8rem;
            font-weight: 800;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: block;
            transition: all var(--transition-normal);
        }}

        .test-stat:hover .test-stat-value {{
            transform: scale(1.05);
        }}

        .test-stat-label {{
            color: var(--text);
            font-size: 0.9rem;
            opacity: 0.9;
            margin-top: 0.5rem;
        }}

        /* Premium Advanced Analysis Styles */
        .analysis-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }}

        .analysis-card {{
            background: var(--glass);
            padding: 2rem;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            backdrop-filter: var(--blur-md);
            transition: all var(--transition-normal);
            position: relative;
            overflow: hidden;
        }}

        .analysis-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-glass);
            pointer-events: none;
        }}

        .analysis-card:hover {{
            transform: translateY(-5px) scale(1.02);
            box-shadow: var(--shadow-hover);
        }}

        .analysis-card h4 {{
            color: var(--primary);
            margin-bottom: 1.2rem;
            font-size: 1.2rem;
            position: relative;
        }}

        .circle-score {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: conic-gradient(var(--success) 0% var(--score-percent), var(--border) var(--score-percent) 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1.2rem;
            position: relative;
            box-shadow: var(--shadow-lg);
            transition: all var(--transition-normal);
        }}

        .analysis-card:hover .circle-score {{
            transform: scale(1.05) rotate(5deg);
        }}

        .circle-score-inner {{
            width: 90px;
            height: 90px;
            border-radius: 50%;
            background: var(--card-bg);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.5rem;
            color: var(--text);
            box-shadow: var(--shadow-sm);
        }}

        .knowledge-gaps {{
            list-style: none;
        }}

        .knowledge-gaps li {{
            padding: 0.8rem;
            margin-bottom: 0.6rem;
            background: var(--glass-dark);
            border-radius: var(--radius-sm);
            border-left: 3px solid var(--warning);
            transition: all var(--transition-normal);
            backdrop-filter: var(--blur-sm);
        }}

        .knowledge-gaps li:hover {{
            transform: translateX(5px);
            box-shadow: var(--shadow-md);
        }}

        .memory-decay-item {{
            padding: 0.8rem;
            margin-bottom: 0.6rem;
            background: var(--glass-dark);
            border-radius: var(--radius-sm);
            border-left: 3px solid var(--danger);
            transition: all var(--transition-normal);
            backdrop-filter: var(--blur-sm);
        }}

        .memory-decay-item:hover {{
            transform: translateX(5px);
            box-shadow: var(--shadow-md);
        }}

        .improvement-suggestion {{
            padding: 0.8rem;
            margin-bottom: 0.6rem;
            background: var(--glass-dark);
            border-radius: var(--radius-sm);
            border-left: 3px solid var(--success);
            transition: all var(--transition-normal);
            backdrop-filter: var(--blur-sm);
        }}

        .improvement-suggestion:hover {{
            transform: translateX(5px);
            box-shadow: var(--shadow-md);
        }}

        /* Premium Rest of the enhanced styles continue... */
        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 0.8rem;
            position: relative;
        }}

        .panel-title {{
            font-size: 1.8rem;
            font-weight: 800;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .stat-card {{
            background: var(--glass);
            padding: 2rem;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            text-align: center;
            backdrop-filter: var(--blur-md);
            transition: all var(--transition-normal);
            position: relative;
            overflow: hidden;
        }}

        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-glass);
            pointer-events: none;
        }}

        .stat-card:hover {{
            transform: translateY(-5px) scale(1.03);
            box-shadow: var(--shadow-hover);
        }}

        .stat-value {{
            font-size: 2.5rem;
            font-weight: 900;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: block;
            line-height: 1;
            transition: all var(--transition-normal);
        }}

        .stat-card:hover .stat-value {{
            transform: scale(1.05);
        }}

        .stat-label {{
            color: var(--text);
            font-size: 0.95rem;
            opacity: 0.9;
            margin-top: 0.6rem;
            transition: all var(--transition-normal);
        }}

        .concepts-grid {{
            display: grid;
            gap: 1.2rem;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        }}

        .concept-card {{
            background: var(--glass);
            padding: 2rem;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            border-left: 4px solid var(--primary);
            transition: all var(--transition-normal);
            backdrop-filter: var(--blur-md);
            position: relative;
            overflow: hidden;
        }}

        .concept-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-glass);
            pointer-events: none;
        }}

        .concept-card:hover {{
            transform: translateY(-5px) scale(1.02);
            box-shadow: var(--shadow-hover);
        }}

        .concept-importance {{
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 15px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: white;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: var(--shadow-sm);
        }}

        .importance-high {{ background: var(--danger); }}
        .importance-medium {{ background: var(--warning); }}
        .importance-low {{ background: var(--success); }}

        .flashcard-container {{
            perspective: 1000px;
            height: 300px;
            margin-bottom: 2rem;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
        }}

        .flashcard {{
            width: 100%;
            height: 100%;
            position: relative;
            transform-style: preserve-3d;
            transition: transform 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
            cursor: pointer;
        }}

        .flashcard.flipped {{
            transform: rotateY(180deg);
        }}

        .flashcard-face {{
            position: absolute;
            width: 100%;
            height: 100%;
            backface-visibility: hidden;
            background: var(--glass);
            border-radius: var(--radius);
            padding: 2rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-xl);
            backdrop-filter: var(--blur-lg);
            transition: all var(--transition-normal);
        }}

        .flashcard-back {{
            transform: rotateY(180deg);
        }}

        .flashcard-nav {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1rem;
            margin-top: 2rem;
            flex-wrap: wrap;
        }}

        .nav-btn {{
            background: var(--gradient-main);
            color: white;
            border: none;
            padding: 1rem 1.5rem;
            border-radius: var(--radius);
            cursor: pointer;
            transition: all var(--transition-normal);
            font-weight: 600;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }}

        .nav-btn:hover {{
            transform: translateY(-2px) scale(1.05);
            box-shadow: var(--shadow-hover);
        }}

        .nav-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }}

        .mcq-container {{
            display: grid;
            gap: 1.5rem;
        }}

        .mcq-card {{
            background: var(--glass);
            padding: 2rem;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            backdrop-filter: var(--blur-md);
            transition: all var(--transition-normal);
            position: relative;
            overflow: hidden;
        }}

        .mcq-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-glass);
            pointer-events: none;
        }}

        .mcq-card:hover {{
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
        }}

        .options-container {{
            display: grid;
            gap: 0.8rem;
            margin: 1.2rem 0;
        }}

        .option-item {{
            padding: 1rem 1.2rem;
            background: var(--glass-dark);
            border-radius: var(--radius-sm);
            cursor: pointer;
            transition: all var(--transition-normal);
            border: 2px solid transparent;
            font-weight: 500;
            backdrop-filter: var(--blur-sm);
            position: relative;
            overflow: hidden;
        }}

        .option-item::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: var(--gradient-main);
            transition: left var(--transition-normal);
            opacity: 0.1;
        }}

        .option-item:hover {{
            border-color: var(--primary);
            transform: translateX(5px);
            box-shadow: var(--shadow-md);
        }}

        .option-item:hover::before {{
            left: 0;
        }}

        .option-item.correct {{
            background: var(--success);
            color: white;
            border-color: var(--success);
            transform: translateX(5px);
            box-shadow: var(--shadow-md);
        }}

        .option-item.incorrect {{
            background: var(--danger);
            color: white;
            border-color: var(--danger);
            transform: translateX(5px);
            box-shadow: var(--shadow-md);
        }}

        .mcq-explanation {{
            margin-top: 1.2rem;
            padding: 1.2rem;
            background: var(--glass-dark);
            border-left: 4px solid var(--primary);
            border-radius: var(--radius-sm);
            display: none;
            animation: fadeIn 0.5s ease;
            backdrop-filter: var(--blur-sm);
            transition: all var(--transition-normal);
        }}

        .mcq-explanation:hover {{
            transform: translateX(3px);
            box-shadow: var(--shadow-md);
        }}

        .mind-map-container {{
            background: var(--glass);
            border-radius: var(--radius);
            padding: 2rem;
            border: 1px solid var(--border);
            min-height: 400px;
            backdrop-filter: var(--blur-lg);
            transition: all var(--transition-normal);
            position: relative;
            overflow: hidden;
        }}

        .mind-map-container::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-glass);
            pointer-events: none;
        }}

        .mind-map-container:hover {{
            transform: translateY(-3px);
            box-shadow: var(--shadow-xl);
        }}

        .mind-map-visual {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2rem;
        }}

        .central-topic {{
            background: var(--gradient-main);
            color: white;
            padding: 1.5rem 2.5rem;
            border-radius: var(--radius);
            font-size: 1.5rem;
            font-weight: 800;
            text-align: center;
            box-shadow: var(--shadow-xl);
            transition: all var(--transition-normal);
        }}

        .central-topic:hover {{
            transform: scale(1.03);
            box-shadow: var(--shadow-hover);
        }}

        .main-branches {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            width: 100%;
        }}

        .branch {{
            background: var(--glass-dark);
            padding: 1.5rem;
            border-radius: var(--radius);
            border-left: 4px solid var(--accent);
            box-shadow: var(--shadow-lg);
            transition: all var(--transition-normal);
            backdrop-filter: var(--blur-sm);
        }}

        .branch:hover {{
            transform: translateY(-5px) scale(1.02);
            box-shadow: var(--shadow-hover);
        }}

        .branch-title {{
            font-weight: 700;
            margin-bottom: 1rem;
            color: var(--accent);
            font-size: 1.2rem;
        }}

        .sub-branches {{
            list-style: none;
            padding-left: 0.8rem;
        }}

        .sub-branches li {{
            margin: 0.5rem 0;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border);
            transition: all var(--transition-normal);
        }}

        .sub-branches li:hover {{
            transform: translateX(3px);
            color: var(--accent);
        }}

        .tricks-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }}

        .trick-card {{
            background: var(--glass);
            padding: 2rem;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            transition: all var(--transition-normal);
            backdrop-filter: var(--blur-md);
            position: relative;
            overflow: hidden;
        }}

        .trick-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-glass);
            pointer-events: none;
        }}

        .trick-card:hover {{
            transform: translateY(-5px) scale(1.02);
            box-shadow: var(--shadow-hover);
        }}

        .trick-icon {{
            font-size: 2.5rem;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1.2rem;
            text-align: center;
            transition: all var(--transition-normal);
        }}

        .trick-card:hover .trick-icon {{
            transform: scale(1.1) translateY(-3px);
        }}

        .history-grid {{
            display: grid;
            gap: 1.2rem;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        }}

        .history-item {{
            background: var(--glass);
            padding: 2rem;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            cursor: pointer;
            transition: all var(--transition-normal);
            backdrop-filter: var(--blur-md);
            position: relative;
            overflow: hidden;
        }}

        .history-item::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-glass);
            pointer-events: none;
        }}

        .history-item:hover {{
            transform: translateY(-5px) scale(1.02);
            border-color: var(--primary);
            box-shadow: var(--shadow-hover);
        }}

        .history-stats {{
            display: flex;
            gap: 0.8rem;
            margin-top: 1rem;
            font-size: 0.85rem;
        }}

        .history-stat {{
            background: var(--primary);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 15px;
            font-weight: 600;
            box-shadow: var(--shadow-sm);
            transition: all var(--transition-normal);
        }}

        .history-item:hover .history-stat {{
            transform: scale(1.05);
        }}

        /* Premium Goals/To-do List Styles */
        .goals-container {{
            background: var(--glass);
            border-radius: var(--radius);
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
            backdrop-filter: var(--blur-lg);
            position: relative;
            overflow: hidden;
        }}

        .goals-container::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-glass);
            pointer-events: none;
        }}

        .goal-form {{
            display: grid;
            gap: 1.2rem;
            margin-bottom: 2rem;
            grid-template-columns: 1fr 1fr;
        }}

        .form-group {{
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }}

        .form-group.full-width {{
            grid-column: 1 / -1;
        }}

        .form-input {{
            padding: 1rem 1.2rem;
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            background: var(--glass-dark);
            color: var(--text);
            font-size: 0.95rem;
            transition: all var(--transition-normal);
            backdrop-filter: var(--blur-sm);
        }}

        .form-input:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
            transform: translateY(-1px);
        }}

        .goal-list {{
            display: grid;
            gap: 1.2rem;
        }}

        .goal-item {{
            background: var(--glass-dark);
            padding: 1.5rem;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 1.2rem;
            transition: all var(--transition-normal);
            backdrop-filter: var(--blur-sm);
            position: relative;
            overflow: hidden;
        }}

        .goal-item::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: var(--gradient-main);
            transition: left var(--transition-normal);
            opacity: 0.1;
        }}

        .goal-item:hover {{
            transform: translateX(5px);
            box-shadow: var(--shadow-lg);
        }}

        .goal-item:hover::before {{
            left: 0;
        }}

        .goal-item.completed {{
            opacity: 0.8;
            background: var(--success);
            color: white;
        }}

        .goal-item.completed .goal-title {{
            text-decoration: line-through;
        }}

        .goal-checkbox {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid var(--border);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all var(--transition-normal);
            background: var(--glass);
            backdrop-filter: var(--blur-sm);
        }}

        .goal-item.completed .goal-checkbox {{
            background: white;
            border-color: white;
        }}

        .goal-content {{
            flex: 1;
        }}

        .goal-title {{
            font-weight: 700;
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }}

        .goal-description {{
            font-size: 0.9rem;
            opacity: 0.9;
            margin-bottom: 0.5rem;
        }}

        .goal-meta {{
            display: flex;
            gap: 0.8rem;
            font-size: 0.8rem;
        }}

        .goal-priority {{
            padding: 0.3rem 0.6rem;
            border-radius: 12px;
            font-weight: 700;
        }}

        .priority-high {{ background: var(--danger); color: white; }}
        .priority-medium {{ background: var(--warning); color: white; }}
        .priority-low {{ background: var(--success); color: white; }}

        .goal-actions {{
            display: flex;
            gap: 0.5rem;
        }}

        .goal-action {{
            background: none;
            border: none;
            color: var(--text);
            cursor: pointer;
            padding: 0.5rem;
            border-radius: 4px;
            transition: all var(--transition-normal);
            backdrop-filter: var(--blur-sm);
        }}

        .goal-action:hover {{
            background: var(--primary);
            color: white;
            transform: scale(1.05);
        }}

        /* Premium Focus Mode Styles */
        .focus-container {{
            background: var(--glass);
            border-radius: var(--radius);
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
            backdrop-filter: var(--blur-lg);
            text-align: center;
            position: relative;
            overflow: hidden;
        }}

        .focus-container::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-glass);
            pointer-events: none;
        }}

        .focus-timer {{
            font-size: 4rem;
            font-weight: 900;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 2rem 0;
            text-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: all var(--transition-normal);
        }}

        .focus-controls {{
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .focus-btn {{
            padding: 1.2rem 2.5rem;
            border: none;
            border-radius: var(--radius);
            font-weight: 700;
            cursor: pointer;
            transition: all var(--transition-normal);
            font-size: 1rem;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }}

        .focus-start {{
            background: var(--gradient-main);
            color: white;
        }}

        .focus-pause {{
            background: var(--warning);
            color: white;
        }}

        .focus-reset {{
            background: var(--danger);
            color: white;
        }}

        .focus-btn:hover {{
            transform: translateY(-3px) scale(1.05);
            box-shadow: var(--shadow-hover);
        }}

        .focus-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-top: 2rem;
        }}

        .focus-stat {{
            background: var(--glass-dark);
            padding: 1.5rem;
            border-radius: var(--radius);
            text-align: center;
            backdrop-filter: var(--blur-sm);
            transition: all var(--transition-normal);
        }}

        .focus-stat:hover {{
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
        }}

        .focus-stat-value {{
            font-size: 2rem;
            font-weight: 800;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: block;
            transition: all var(--transition-normal);
        }}

        .focus-stat:hover .focus-stat-value {{
            transform: scale(1.05);
        }}

        .focus-stat-label {{
            color: var(--text);
            font-size: 0.95rem;
            opacity: 0.9;
            margin-top: 0.6rem;
        }}

        /* Focus Mode Blur Effect */
        .focus-blur {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            backdrop-filter: blur(15px);
            background: rgba(0, 0, 0, 0.7);
            z-index: 9998;
            display: none;
        }}

        .focus-mode-active {{
            overflow: hidden;
        }}

        .focus-mode-active .focus-blur {{
            display: block;
        }}

        .focus-mode-active .app-container {{
            filter: blur(3px);
        }}

        .focus-exit-btn {{
            position: fixed;
            top: 1.5rem;
            right: 1.5rem;
            background: var(--danger);
            color: white;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: var(--radius);
            font-weight: 700;
            cursor: pointer;
            z-index: 9999;
            box-shadow: var(--shadow-xl);
            transition: all var(--transition-normal);
        }}

        .focus-exit-btn:hover {{
            transform: scale(1.05);
            box-shadow: var(--shadow-hover);
        }}

        .loading-spinner {{
            display: inline-block;
            width: 50px;
            height: 50px;
            border: 3px solid var(--border);
            border-left: 3px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            box-shadow: var(--shadow-sm);
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(15px) scale(0.95); }}
            to {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}

        .text-center {{ text-align: center; }}
        .text-muted {{ color: var(--muted); }}
        .mt-2 {{ margin-top: 1rem; }}
        .mb-2 {{ margin-bottom: 1rem; }}

        /* Premium Responsive Design */
        @media (max-width: 1200px) {{
            .content-area {{
                padding: 1.5rem;
            }}
            
            .sidebar {{
                width: 250px;
            }}
            
            .sidebar.collapsed {{
                width: 70px;
            }}
        }}

        @media (max-width: 768px) {{
            .sidebar {{
                transform: translateX(-100%);
            }}
            
            .sidebar.mobile-open {{
                transform: translateX(0);
            }}
            
            .main-content {{
                margin-left: 0;
            }}
            
            .main-content.expanded {{
                margin-left: 0;
            }}
            
            .content-area {{
                padding: 1rem;
            }}
            
            .top-header {{
                padding: 0 1rem;
            }}
            
            .header h1 {{
                font-size: 2rem;
            }}
            
            .concepts-grid {{
                grid-template-columns: 1fr;
            }}
            
            .tricks-grid {{
                grid-template-columns: 1fr;
            }}
            
            .chatbot-window {{
                width: 95vw;
                height: 60vh;
                right: 2.5vw;
            }}
            
            .upload-section {{
                padding: 1.5rem;
            }}
            
            .upload-area {{
                padding: 2rem 1.5rem;
            }}
            
            .analysis-grid {{
                grid-template-columns: 1fr;
            }}
            
            .goal-form {{
                grid-template-columns: 1fr;
            }}
            
            .focus-controls {{
                flex-direction: column;
                align-items: center;
            }}
            
            .focus-btn {{
                width: 180px;
            }}
        }}

        @media (max-width: 480px) {{
            .header h1 {{
                font-size: 1.8rem;
            }}
            
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
            
            .content-tabs {{
                flex-direction: column;
            }}
            
            .tab-btn {{
                min-width: auto;
            }}
            
            .chatbot-toggle {{
                width: 60px;
                height: 60px;
                font-size: 1.5rem;
            }}
            
            .focus-timer {{
                font-size: 3rem;
            }}
        }}

        /* Enhanced MCQ Test Interface */
        .test-interface {{
            background: var(--glass);
            border-radius: var(--radius);
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
            backdrop-filter: var(--blur-lg);
            position: relative;
            overflow: hidden;
        }}

        .test-interface::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--gradient-glass);
            pointer-events: none;
        }}

        .test-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--border);
        }}

        .test-timer {{
            font-size: 1.2rem;
            font-weight: 700;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 0.5rem 1rem;
            border-radius: var(--radius-sm);
            border: 2px solid var(--primary);
        }}

        .test-progress {{
            display: flex;
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 1.5rem;
        }}

        .progress-bar {{
            flex: 1;
            height: 6px;
            background: var(--border);
            border-radius: 3px;
            overflow: hidden;
        }}

        .progress-fill {{
            height: 100%;
            background: var(--gradient-main);
            border-radius: 3px;
            transition: width 0.3s ease;
        }}

        .question-navigation {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 2px solid var(--border);
        }}

        .nav-buttons {{
            display: flex;
            gap: 0.8rem;
        }}

        .question-number {{
            font-weight: 700;
            color: var(--primary);
            font-size: 0.95rem;
        }}

        .submit-test-btn {{
            background: var(--success);
            color: white;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: var(--radius-sm);
            font-weight: 600;
            cursor: pointer;
            transition: all var(--transition-normal);
        }}

        .submit-test-btn:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }}

        /* Enhanced Analysis Dashboard */
        .analysis-dashboard {{
            background: var(--glass);
            border-radius: var(--radius);
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
            backdrop-filter: var(--blur-lg);
        }}

        .score-circle {{
            width: 160px;
            height: 160px;
            margin: 0 auto 1.5rem;
            position: relative;
        }}

        .circle-bg {{
            fill: none;
            stroke: var(--border);
            stroke-width: 8;
        }}

        .circle-progress {{
            fill: none;
            stroke: var(--success);
            stroke-width: 8;
            stroke-linecap: round;
            transform: rotate(-90deg);
            transform-origin: 50% 50%;
            transition: stroke-dasharray 1s ease;
        }}

        .circle-text {{
            font-size: 2rem;
            font-weight: 900;
            fill: var(--text);
        }}

        .circle-label {{
            font-size: 0.9rem;
            fill: var(--muted);
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }}

        .metric-card {{
            background: var(--glass-dark);
            padding: 1.2rem;
            border-radius: var(--radius-sm);
            text-align: center;
            backdrop-filter: var(--blur-sm);
        }}

        .metric-value {{
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--primary);
        }}

        .metric-label {{
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }}

        .charts-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }}

        .chart-card {{
            background: var(--glass-dark);
            padding: 1.5rem;
            border-radius: var(--radius-sm);
            backdrop-filter: var(--blur-sm);
        }}

        .recommendations {{
            background: var(--glass-dark);
            padding: 1.5rem;
            border-radius: var(--radius-sm);
            margin-top: 1.5rem;
            backdrop-filter: var(--blur-sm);
        }}

        .recommendation-item {{
            padding: 0.8rem;
            margin-bottom: 0.8rem;
            background: var(--glass);
            border-radius: var(--radius-sm);
            border-left: 3px solid var(--info);
        }}
    </style>
</head>
<body class="theme-{user_settings['theme']}">
    <!-- Focus Mode Blur Overlay -->
    <div class="focus-blur" id="focus-blur"></div>
    <button class="focus-exit-btn" id="focus-exit-btn" style="display: none;" onclick="exitFocusMode()">
        <i class="fas fa-times"></i> Exit Focus Mode
    </button>

    <div class="app-container">
        <!-- Premium Enhanced Sidebar Navigation -->
        <nav class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <div class="logo">
                    <i class="fas fa-robot"></i>
                    <span>Study Suite Pro</span>
                </div>
                <button class="toggle-sidebar" onclick="toggleSidebar()">
                    <i class="fas fa-bars"></i>
                </button>
            </div>
            
            <div class="sidebar-content">
                <div class="nav-section">
                    <div class="nav-title">Main Navigation</div>
                    <div class="nav-items">
                        <div class="nav-item active" onclick="switchPanel('generator')">
                            <div class="nav-icon"><i class="fas fa-magic"></i></div>
                            <div class="nav-text">AI Generator</div>
                        </div>
                        <div class="nav-item" onclick="switchPanel('daily-test')">
                            <div class="nav-icon"><i class="fas fa-calendar-day"></i></div>
                            <div class="nav-text">Daily Revision</div>
                        </div>
                        <div class="nav-item" onclick="switchPanel('history')">
                            <div class="nav-icon"><i class="fas fa-history"></i></div>
                            <div class="nav-text">Study History</div>
                        </div>
                    </div>
                </div>
                
                <div class="nav-section">
                    <div class="nav-title">Study Tools</div>
                    <div class="nav-items">
                        <div class="nav-item" onclick="switchPanel('formulas')">
                            <div class="nav-icon"><i class="fas fa-square-root-variable"></i></div>
                            <div class="nav-text">Formulas</div>
                        </div>
                        <div class="nav-item" onclick="switchPanel('goals')">
                            <div class="nav-icon"><i class="fas fa-bullseye"></i></div>
                            <div class="nav-text">Goals</div>
                        </div>
                        <div class="nav-item" onclick="switchPanel('focus')">
                            <div class="nav-icon"><i class="fas fa-hourglass-half"></i></div>
                            <div class="nav-text">Focus Mode</div>
                        </div>
                    </div>
                </div>
                
                <div class="theme-selector-container">
                    <label class="theme-selector-label">Theme Selection</label>
                    <select class="theme-selector" id="theme-selector" onchange="changeTheme(this.value)">
                        <option value="dark" {'selected' if user_settings['theme'] == 'dark' else ''}>🌙 Dark Mode</option>
                        <option value="light" {'selected' if user_settings['theme'] == 'light' else ''}>☀️ Light Mode</option>
                        <option value="ocean" {'selected' if user_settings['theme'] == 'ocean' else ''}>🌊 Ocean Blue</option>
                        <option value="forest" {'selected' if user_settings['theme'] == 'forest' else ''}>🌲 Forest Green</option>
                        <option value="fire" {'selected' if user_settings['theme'] == 'fire' else ''}>🔥 Fire Red</option>
                        <option value="cosmic" {'selected' if user_settings['theme'] == 'cosmic' else ''}>🌌 Cosmic Purple</option>
                        <option value="sunset" {'selected' if user_settings['theme'] == 'sunset' else ''}>🌅 Sunset Orange</option>
                        <option value="midnight" {'selected' if user_settings['theme'] == 'midnight' else ''}>🌃 Midnight Blue</option>
                        <option value="synthwave" {'selected' if user_settings['theme'] == 'synthwave' else ''}>🎮 Synthwave</option>
                        <option value="emerald" {'selected' if user_settings['theme'] == 'emerald' else ''}>💎 Emerald</option>
                        <option value="volcano" {'selected' if user_settings['theme'] == 'volcano' else ''}>🌋 Volcano</option>
                    </select>
                </div>
            </div>
        </nav>

        <div class="main-content" id="main-content">
            <header class="top-header">
                <h1 class="page-title" id="page-title">AI Study Generator Pro</h1>
                <div class="header-controls">
                    <div class="productivity-display">
                        <span class="text-muted">Enhanced Learning Suite</span>
                    </div>
                </div>
            </header>

            <div class="content-area">
                <!-- AI Generator Panel -->
                <div id="generator-panel" class="content-panel active">
                    <div class="header">
                        <h1>AI Study Suite Pro</h1>
                        <p>Transform your study materials into comprehensive learning resources with AI-powered SRbot</p>
                    </div>

                    <div class="upload-section">
                        <div class="upload-area" onclick="document.getElementById('file-input').click()">
                            <i class="fas fa-cloud-upload-alt"></i>
                            <h3>Upload Study Materials</h3>
                            <p>Click to upload PDF files (Supports files up to 100MB)</p>
                            <span id="file-name" class="text-muted">No file selected</span>
                        </div>
                        
                        <input type="file" id="file-input" class="file-input" accept=".pdf,.txt">
                        
                        <button id="generate-btn" class="generate-btn" onclick="generateContent()">
                            <i class="fas fa-bolt"></i>
                            Generate Study Pack with SRbot
                        </button>
                    </div>

                    <div id="loading-section" style="display: none; text-align: center; padding: 3rem;">
                        <div class="loading-spinner"></div>
                        <p style="margin-top: 1.5rem; font-size: 1.1rem; color: var(--text);">SRbot is crafting your personalized study materials... This may take just 1 minute</p>
                    </div>

                    <div id="content-section" style="display: none;">
                        <div class="mcq-performance" id="mcq-performance">
                            <div class="performance-header">
                                <h2 class="panel-title">MCQ Performance Report</h2>
                                <button class="nav-btn" onclick="resetMCQs()">
                                    <i class="fas fa-redo"></i> Try Again
                                </button>
                            </div>
                            
                            <div class="score-display">
                                <span class="score-value" id="score-value"></span>
                                <span id="score-message">Complete the MCQs to see your score</span>
                            </div>
                            
                            <!-- Enhanced MCQ Analysis -->
                            <div class="mcq-analysis-grid" id="mcq-analysis-grid" style="display: none;">
                                <div class="analysis-dashboard">
                                    <div class="score-circle">
                                        <svg width="160" height="160" viewBox="0 0 160 160">
                                            <circle class="circle-bg" cx="80" cy="80" r="70"></circle>
                                            <circle class="circle-progress" cx="80" cy="80" r="70" id="performance-circle"></circle>
                                            <text class="circle-text" x="80" y="85" text-anchor="middle" id="circle-score">0%</text>
                                            <text class="circle-label" x="80" y="105" text-anchor="middle">Score</text>
                                        </svg>
                                    </div>
                                    
                                    <div class="metrics-grid">
                                        <div class="metric-card">
                                            <div class="metric-value" id="total-questions">0</div>
                                            <div class="metric-label">Total Questions</div>
                                        </div>
                                        <div class="metric-card">
                                            <div class="metric-value" id="correct-answers">0</div>
                                            <div class="metric-label">Correct Answers</div>
                                        </div>
                                        <div class="metric-card">
                                            <div class="metric-value" id="incorrect-answers">0</div>
                                            <div class="metric-label">Incorrect Answers</div>
                                        </div>
                                        <div class="metric-card">
                                            <div class="metric-value" id="time-taken">0s</div>
                                            <div class="metric-label">Time Taken</div>
                                        </div>
                                    </div>

                                    <div class="charts-container">
                                        <div class="chart-card">
                                            <canvas id="difficultyChart" height="200"></canvas>
                                        </div>
                                        <div class="chart-card">
                                            <canvas id="categoryChart" height="200"></canvas>
                                        </div>
                                    </div>

                                    <div class="recommendations">
                                        <h4><i class="fas fa-lightbulb"></i> Personalized Recommendations</h4>
                                        <div id="recommendations-list"></div>
                                    </div>
                                </div>
                            </div>

                            <div class="weak-areas">
                                <h4><i class="fas fa-exclamation-triangle"></i> Areas to Improve</h4>
                                <ul class="area-list" id="weak-areas-list"></ul>
                            </div>

                            <div class="weak-areas" style="border-left-color: var(--success);">
                                <h4><i class="fas fa-star"></i> Your Strengths</h4>
                                <ul class="area-list" id="strong-areas-list"></ul>
                            </div>

                            <button class="detailed-analysis-btn" onclick="toggleDetailedAnalysis()">
                                <i class="fas fa-chart-bar"></i> Show Detailed Analysis
                            </button>

                            <div class="detailed-analysis" id="detailed-analysis">
                                <div class="analysis-section">
                                    <h4><i class="fas fa-trophy"></i> Overall Performance</h4>
                                    <div id="overall-performance"></div>
                                </div>
                                <div class="analysis-section">
                                    <h4><i class="fas fa-bug"></i> Mistakes Analysis</h4>
                                    <div id="mistakes-analysis"></div>
                                </div>
                                <div class="analysis-section">
                                    <h4><i class="fas fa-lightbulb"></i> Improvement Suggestions</h4>
                                    <div id="improvement-suggestions"></div>
                                </div>
                                <div class="analysis-section">
                                    <h4><i class="fas fa-star"></i> Strengths</h4>
                                    <div id="strengths-analysis"></div>
                                </div>
                            </div>
                        </div>

                        <div class="stats-grid">
                            <div class="stat-card">
                                <span class="stat-value" id="concepts-count"></span>
                                <span class="stat-label">Key Concepts</span>
                            </div>
                            <div class="stat-card">
                                <span class="stat-value" id="flashcards-count"></span>
                                <span class="stat-label">Flashcards</span>
                            </div>
                            <div class="stat-card">
                                <span class="stat-value" id="mcqs-count"></span>
                                <span class="stat-label">MCQs (30-40)</span>
                            </div>
                            <div class="stat-card">
                                <span class="stat-value" id="tricks-count"></span>
                                <span class="stat-label">Memory Techniques</span>
                            </div>
                        </div>

                        <div class="content-tabs">
                            <button class="tab-btn active" onclick="switchTab('concepts')">Key Concepts</button>
                            <button class="tab-btn" onclick="switchTab('flashcards')">Flashcards</button>
                            <button class="tab-btn" onclick="switchTab('mcqs')">MCQs</button>
                            <button class="tab-btn" onclick="switchTab('mindmap')">Mind Map</button>
                            <button class="tab-btn" onclick="switchTab('tricks')">Memory Tricks</button>
                        </div>

                        <div id="concepts-panel" class="content-panel active">
                            <div class="panel-header">
                                <h2 class="panel-title">Key Concepts</h2>
                                <div class="text-muted" id="concepts-summary"></div>
                            </div>
                            <div id="concepts-content" class="concepts-grid"></div>
                        </div>

                        <div id="flashcards-panel" class="content-panel">
                            <div class="panel-header">
                                <h2 class="panel-title">Interactive Flashcards</h2>
                                <div class="text-muted" id="flashcards-summary"></div>
                            </div>
                            <div id="flashcards-content"></div>
                        </div>

                        <div id="mcqs-panel" class="content-panel">
                            <div class="panel-header">
                                <h2 class="panel-title">Multiple Choice Questions</h2>
                                <div class="text-muted" id="mcqs-summary"></div>
                            </div>
                            <div class="test-controls" style="margin-bottom: 1.5rem; display: none;" id="test-controls">
                                <button class="generate-btn" onclick="startTestSession('session')">
                                    <i class="fas fa-play"></i> Start Test Session
                                </button>
                            </div>
                            <div id="mcqs-content" class="mcq-container"></div>
                            <button id="mcq-analysis-btn" class="generate-btn" style="display:none; margin-top: 1.5rem;" onclick="showPerformanceReport()">
                                <i class="fas fa-chart-bar"></i> View Enhanced Analysis
                            </button>
                        </div>

                        <div id="mindmap-panel" class="content-panel">
                            <div class="panel-header">
                                <h2 class="panel-title">Visual Mind Map</h2>
                                <div class="text-muted">Interactive concept relationships</div>
                            </div>
                            <div id="mindmap-content" class="mind-map-container"></div>
                        </div>

                        <div id="tricks-panel" class="content-panel">
                            <div class="panel-header">
                                <h2 class="panel-title">Memory Enhancement Tricks</h2>
                                <div class="text-muted">Techniques to improve retention</div>
                            </div>
                            <div id="tricks-content" class="tricks-grid"></div>
                        </div>
                    </div>
                </div>

                <!-- Daily Test Panel -->
                <div id="daily-test-panel" class="content-panel">
                    <div class="header">
                        <h1>Daily Revision Test</h1>
                        <p>Start Daily Prep with 20+ adaptive questions combining all your uploaded PDFs for guaranteed revision</p>
                    </div>

                    <div class="daily-test-section">
                        <div class="test-header">
                            <h2 class="test-title">Today's Revision Test</h2>
                            <button class="generate-btn" onclick="startTestSession('daily')" id="start-test-btn">
                                <i class="fas fa-play"></i> Start Daily Prep
                            </button>
                        </div>
                        
                        <div class="test-description">
                            <p><strong>"Start Daily Prep" is your core feature for guaranteed revision.</strong> The AI generates 20+ adaptive questions daily, combining all uploaded PDFs into a single, high-impact review set. After the non-mandatory test, deep analysis pinpoints Critical Knowledge Gaps and diagnoses Memory Decay, ensuring every minute spent actually reinforces long-term retention.</p>
                        </div>

                        <div class="test-stats">
                            <div class="test-stat">
                                <span class="test-stat-value" id="test-questions-count">{len(daily_test['questions']) if daily_test and daily_test['questions'] else 0}</span>
                                <span class="test-stat-label">Total Questions</span>
                            </div>
                            <div class="test-stat">
                                <span class="test-stat-value" id="test-completed">{'Yes' if daily_test and daily_test['completed'] else 'No'}</span>
                                <span class="test-stat-label">Completed Today</span>
                            </div>
                            <div class="test-stat">
                                <span class="test-stat-value" id="test-score">{daily_test['score'] if daily_test and daily_test['completed'] else 'N/A'}</span>
                                <span class="test-stat-label">Best Score</span>
                            </div>
                            <div class="test-stat">
                                <span class="test-stat-value" id="test-streak">{len([s for s in test_sessions if s['test_type'] == 'daily'])}</span>
                                <span class="test-stat-label">Tests Taken</span>
                            </div>
                        </div>

                        <!-- Test Interface for Daily Test -->
                        <div class="test-interface" id="daily-test-interface" style="display: none;">
                            <div class="test-header">
                                <h3>Daily Revision Test</h3>
                                <div class="test-timer" id="daily-test-timer">30:00</div>
                            </div>
                            
                            <div class="test-progress">
                                <span>Progress:</span>
                                <div class="progress-bar">
                                    <div class="progress-fill" id="daily-test-progress" style="width: 0%"></div>
                                </div>
                                <span id="daily-test-progress-text">0/0</span>
                            </div>

                            <div class="question-container" id="daily-test-question-container">
                                <div class="question-text" id="daily-test-question-text"></div>
                                <div class="options-container" id="daily-test-options"></div>
                            </div>

                            <div class="question-navigation">
                                <div class="nav-buttons">
                                    <button class="test-control-btn nav-btn" onclick="previousQuestion('daily')" id="daily-prev-btn">
                                        <i class="fas fa-chevron-left"></i> Previous
                                    </button>
                                    <button class="test-control-btn nav-btn" onclick="nextQuestion('daily')" id="daily-next-btn">
                                        Next <i class="fas fa-chevron-right"></i>
                                    </button>
                                </div>
                                <div class="question-number" id="daily-test-question-number">Question 1 of 0</div>
                                <div class="nav-buttons">
                                    <button class="test-control-btn exit-btn" onclick="exitTest('daily')">
                                        <i class="fas fa-times"></i> Exit Test
                                    </button>
                                    <button class="test-control-btn submit-btn" onclick="submitTest('daily')" id="daily-submit-btn" style="display: none;">
                                        <i class="fas fa-paper-plane"></i> Submit Test
                                    </button>
                                </div>
                            </div>
                        </div>

                        <!-- Test Results -->
                        <div id="daily-test-results" style="display: none;">
                            <div class="analysis-dashboard" id="daily-test-analysis">
                                <div class="score-circle">
                                    <svg width="160" height="160" viewBox="0 0 160 160">
                                        <circle class="circle-bg" cx="80" cy="80" r="70"></circle>
                                        <circle class="circle-progress" cx="80" cy="80" r="70" id="daily-score-circle"></circle>
                                        <text class="circle-text" x="80" y="85" text-anchor="middle" id="daily-circle-score">0%</text>
                                        <text class="circle-label" x="80" y="105" text-anchor="middle">Score</text>
                                    </svg>
                                </div>
                                
                                <div class="metrics-grid">
                                    <div class="metric-card">
                                        <div class="metric-value" id="daily-total-questions">0</div>
                                        <div class="metric-label">Total Questions</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value" id="daily-correct-answers">0</div>
                                        <div class="metric-label">Correct Answers</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value" id="daily-incorrect-answers">0</div>
                                        <div class="metric-label">Incorrect Answers</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value" id="daily-time-taken">0s</div>
                                        <div class="metric-label">Time Taken</div>
                                    </div>
                                </div>

                                <div class="charts-container">
                                    <div class="chart-card">
                                        <canvas id="daily-difficultyChart" height="200"></canvas>
                                    </div>
                                    <div class="chart-card">
                                        <canvas id="daily-categoryChart" height="200"></canvas>
                                    </div>
                                </div>

                                <div class="recommendations">
                                    <h4><i class="fas fa-lightbulb"></i> Daily Test Recommendations</h4>
                                    <div id="daily-recommendations-list"></div>
                                </div>

                                <div style="text-align: center; margin-top: 2rem;">
                                    <button class="generate-btn" onclick="retakeTest('daily')" style="margin-right: 1rem;">
                                        <i class="fas fa-redo"></i> Retake Test
                                    </button>
                                    <button class="nav-btn" onclick="switchPanel('generator')">
                                        <i class="fas fa-home"></i> Back to Generator
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Study History Panel -->
                <div id="history-panel" class="content-panel">
                    <div class="header">
                        <h1>Study History</h1>
                        <p>Your previously generated study materials with SRbot</p>
                    </div>
                    <div id="history-content" class="history-grid"></div>
                </div>

                <!-- Formulas Panel -->
                <div id="formulas-panel" class="content-panel">
                    <div class="header">
                        <h1>Formulas & Equations</h1>
                        <p>Mathematical formulas and equations extracted from your documents</p>
                    </div>
                    <div class="upload-section">
                        <div class="upload-area" onclick="document.getElementById('formulas-file-input').click()">
                            <i class="fas fa-square-root-variable"></i>
                            <h3>Upload Document for Formula Extraction</h3>
                            <p>Click to upload PDF files to extract mathematical formulas</p>
                            <span id="formulas-file-name" class="text-muted">No file selected</span>
                        </div>
                        <input type="file" id="formulas-file-input" class="file-input" accept=".pdf,.txt">
                        <button class="generate-btn" onclick="extractFormulas()">
                            <i class="fas fa-calculator"></i> Extract Formulas
                        </button>
                    </div>
                    <div id="formulas-content" class="concepts-grid" style="display: none;">
                        <div class="concept-card">
                            <h3>Extracted Formulas</h3>
                            <div id="formulas-list"></div>
                        </div>
                    </div>
                </div>

                <!-- Goals Panel -->
                <div id="goals-panel" class="content-panel">
                    <div class="header">
                        <h1>Study Goals & To-Do List</h1>
                        <p>Set and track your study goals and objectives</p>
                    </div>
                    
                    <div class="goals-container">
                        <h2 class="panel-title">Create New Goal</h2>
                        <div class="goal-form">
                            <div class="form-group">
                                <label>Goal Title</label>
                                <input type="text" id="goal-title" class="form-input" placeholder="Enter goal title">
                            </div>
                            <div class="form-group">
                                <label>Category</label>
                                <select id="goal-category" class="form-input">
                                    <option value="study">Study</option>
                                    <option value="revision">Revision</option>
                                    <option value="project">Project</option>
                                    <option value="exam">Exam Preparation</option>
                                    <option value="other">Other</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Priority</label>
                                <select id="goal-priority" class="form-input">
                                    <option value="high">High</option>
                                    <option value="medium">Medium</option>
                                    <option value="low">Low</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Due Date</label>
                                <input type="date" id="goal-due-date" class="form-input">
                            </div>
                            <div class="form-group full-width">
                                <label>Description</label>
                                <textarea id="goal-description" class="form-input" placeholder="Enter goal description" rows="3"></textarea>
                            </div>
                            <div class="form-group full-width">
                                <button class="generate-btn" onclick="addGoal()">
                                    <i class="fas fa-plus"></i> Add Goal
                                </button>
                            </div>
                        </div>

                        <div class="stats-grid">
                            <div class="stat-card">
                                <span class="stat-value" id="total-goals">{total_goals}</span>
                                <span class="stat-label">Total Goals</span>
                            </div>
                            <div class="stat-card">
                                <span class="stat-value" id="completed-goals">{completed_goals}</span>
                                <span class="stat-label">Completed</span>
                            </div>
                            <div class="stat-card">
                                <span class="stat-value" id="pending-goals">{pending_goals}</span>
                                <span class="stat-label">Pending</span>
                            </div>
                            <div class="stat-card">
                                <span class="stat-value" id="completion-rate">{round((completed_goals/total_goals)*100, 1) if total_goals > 0 else 0}%</span>
                                <span class="stat-label">Completion Rate</span>
                            </div>
                        </div>

                        <h2 class="panel-title">Your Goals</h2>
                        <div class="goal-list" id="goals-list">
                            {"".join([f'''
                            <div class="goal-item {'completed' if goal['completed'] else ''}" data-id="{goal['id']}">
                                <div class="goal-checkbox" onclick="toggleGoal({goal['id']}, this)">
                                    {'''<i class="fas fa-check" style="color: var(--success);"></i>''' if goal['completed'] else ''}
                                </div>
                                <div class="goal-content">
                                    <div class="goal-title">{goal['title']}</div>
                                    <div class="goal-description">{goal['description']}</div>
                                    <div class="goal-meta">
                                        <span class="goal-priority priority-{goal['priority']}">{goal['priority'].upper()}</span>
                                        <span>{goal['category']}</span>
                                        <span>{goal['due_date'] if goal['due_date'] else 'No due date'}</span>
                                    </div>
                                </div>
                                <div class="goal-actions">
                                    <button class="goal-action" onclick="deleteGoal({goal['id']})">
                                        <i class="fas fa-trash"></i>
                                    </button>
                                </div>
                            </div>
                            ''' for goal in user_goals]) if user_goals else '''
                            <div class="text-center" style="padding: 2rem; color: var(--muted);">
                                <i class="fas fa-bullseye" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                                <p>No goals set yet. Create your first study goal above!</p>
                            </div>
                            '''}
                        </div>
                    </div>
                </div>

                <!-- Focus Mode Panel -->
                <div id="focus-panel" class="content-panel">
                    <div class="header">
                        <h1>Focus Mode</h1>
                        <p>Deep work sessions with Pomodoro technique and productivity tracking</p>
                    </div>
                    
                    <div class="focus-container">
                        <h2 class="panel-title">Focus Timer</h2>
                        <div class="focus-timer" id="focus-timer">25:00</div>
                        
                        <div class="focus-controls">
                            <button class="focus-btn focus-start" onclick="startFocusTimer()">
                                <i class="fas fa-play"></i> Start Focus
                            </button>
                            <button class="focus-btn focus-pause" onclick="pauseFocusTimer()">
                                <i class="fas fa-pause"></i> Pause
                            </button>
                            <button class="focus-btn focus-reset" onclick="resetFocusTimer()">
                                <i class="fas fa-redo"></i> Reset
                            </button>
                        </div>

                        <div class="focus-stats">
                            <div class="focus-stat">
                                <span class="focus-stat-value" id="total-focus-time">{total_focus_time}</span>
                                <span class="focus-stat-label">Total Focus Minutes</span>
                            </div>
                            <div class="focus-stat">
                                <span class="focus-stat-value" id="focus-sessions-count">{len(focus_sessions)}</span>
                                <span class="focus-stat-label">Sessions Completed</span>
                            </div>
                            <div class="focus-stat">
                                <span class="focus-stat-value" id="avg-focus-session">{round(avg_focus_session, 1)}</span>
                                <span class="focus-stat-label">Avg Session (min)</span>
                            </div>
                            <div class="focus-stat">
                                <span class="focus-stat-value" id="focus-streak">{len([s for s in focus_sessions if (date.today() - datetime.fromisoformat(s['created_at']).date()).days <= 1])}</span>
                                <span class="focus-stat-label">Recent Sessions</span>
                            </div>
                        </div>

                        <button class="generate-btn" onclick="enterFocusMode()" style="margin-top: 1.5rem;">
                            <i class="fas fa-enter"></i> Enter Full Focus Mode
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Session Test Interface -->
    <div class="test-interface" id="session-test-interface" style="display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 90%; max-width: 800px; max-height: 90vh; overflow-y: auto; z-index: 1001; background: var(--glass-dark);">
        <div class="test-header">
            <h3>Session Test</h3>
            <div class="test-timer" id="session-test-timer">30:00</div>
        </div>
        
        <div class="test-progress">
            <span>Progress:</span>
            <div class="progress-bar">
                <div class="progress-fill" id="session-test-progress" style="width: 0%"></div>
            </div>
            <span id="session-test-progress-text">0/0</span>
        </div>

        <div class="question-container" id="session-test-question-container">
            <div class="question-text" id="session-test-question-text"></div>
            <div class="options-container" id="session-test-options"></div>
        </div>

        <div class="question-navigation">
            <div class="nav-buttons">
                <button class="test-control-btn nav-btn" onclick="previousQuestion('session')" id="session-prev-btn">
                    <i class="fas fa-chevron-left"></i> Previous
                </button>
                <button class="test-control-btn nav-btn" onclick="nextQuestion('session')" id="session-next-btn">
                    Next <i class="fas fa-chevron-right"></i>
                </button>
            </div>
            <div class="question-number" id="session-test-question-number">Question 1 of 0</div>
            <div class="nav-buttons">
                <button class="test-control-btn exit-btn" onclick="exitTest('session')">
                    <i class="fas fa-times"></i> Exit Test
                </button>
                <button class="test-control-btn submit-btn" onclick="submitTest('session')" id="session-submit-btn" style="display: none;">
                    <i class="fas fa-paper-plane"></i> Submit Test
                </button>
            </div>
        </div>
    </div>

    <!-- Session Test Results -->
    <div class="test-interface" id="session-test-results" style="display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 90%; max-width: 800px; max-height: 90vh; overflow-y: auto; z-index: 1001; background: var(--glass-dark);">
        <div class="analysis-dashboard">
            <div class="score-circle">
                <svg width="160" height="160" viewBox="0 0 160 160">
                    <circle class="circle-bg" cx="80" cy="80" r="70"></circle>
                    <circle class="circle-progress" cx="80" cy="80" r="70" id="session-score-circle"></circle>
                    <text class="circle-text" x="80" y="85" text-anchor="middle" id="session-circle-score">0%</text>
                    <text class="circle-label" x="80" y="105" text-anchor="middle">Score</text>
                </svg>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" id="session-total-questions">0</div>
                    <div class="metric-label">Total Questions</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="session-correct-answers">0</div>
                    <div class="metric-label">Correct Answers</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="session-incorrect-answers">0</div>
                    <div class="metric-label">Incorrect Answers</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="session-time-taken">0s</div>
                    <div class="metric-label">Time Taken</div>
                </div>
            </div>

            <div class="charts-container">
                <div class="chart-card">
                    <canvas id="session-difficultyChart" height="200"></canvas>
                </div>
                <div class="chart-card">
                    <canvas id="session-categoryChart" height="200"></canvas>
                </div>
            </div>

            <div class="recommendations">
                <h4><i class="fas fa-lightbulb"></i> Session Test Recommendations</h4>
                <div id="session-recommendations-list"></div>
            </div>

            <div style="text-align: center; margin-top: 2rem;">
                <button class="generate-btn" onclick="retakeTest('session')" style="margin-right: 1rem;">
                    <i class="fas fa-redo"></i> Retake Test
                </button>
                <button class="nav-btn" onclick="closeSessionResults()">
                    <i class="fas fa-times"></i> Close
                </button>
            </div>
        </div>
    </div>

    <!-- Test Overlay -->
    <div class="focus-blur" id="test-overlay" style="display: none; z-index: 1000;"></div>

    <!-- Premium Enhanced Chat Bot -->
    <div class="chatbot-container">
        <div class="chatbot-window" id="chatbot-window">
            <div class="chatbot-header">
                <h3><i class="fas fa-robot"></i> SRbot Assistant</h3>
                <button class="formula-toggle" onclick="toggleFormulas()">
                    <i class="fas fa-square-root-variable"></i> Formulas
                </button>
            </div>
            
            <div class="formulas-sidebar" id="formulas-sidebar">
                <div class="formulas-header">
                    <h4><i class="fas fa-calculator"></i> Document Formulas</h4>
                    <button class="back-button" onclick="toggleFormulas()">
                        <i class="fas fa-arrow-left"></i> Back
                    </button>
                </div>
                <div id="formulas-list"></div>
            </div>
            
            <div class="chatbot-messages" id="chatbot-messages">
                <div class="message bot">
                    <strong>SRbot:</strong> Hello! I'm your AI study assistant. I can help you with study materials, answer questions about this app, or assist with general study techniques. What would you like to know?
                    <div class="message-actions">
                        <button class="message-action" onclick="speakMessage(this.parentElement.parentElement)">
                            <i class="fas fa-volume-up"></i> Speak
                        </button>
                        <button class="message-action" onclick="stopAudio()" style="display: none;" id="stop-audio-btn">
                            <i class="fas fa-stop"></i> Stop
                        </button>
                    </div>
                </div>
            </div>
            
            <div class="chatbot-input">
                <input type="text" id="chat-input" placeholder="Ask SRbot anything about studying or this app..." onkeypress="handleChatKeypress(event)">
                <button onclick="sendMessage()"><i class="fas fa-paper-plane"></i></button>
            </div>
        </div>
        
        <button class="chatbot-toggle" onclick="toggleChatbot()"></button>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Global variables
        let currentContent = null;
        let currentFlashcardIndex = 0;
        let currentFormulas = [];
        let currentDocumentText = '';
        let chatHistory = [];
        let currentTheme = '{user_settings['theme']}';
        let mcqAnswers = [];
        let mcqResults = {{
            total: 0,
            correct: 0,
            weakAreas: [],
            strongAreas: [],
            detailedAnalysis: {{}}
        }};
        let userSettings = {{
            voiceEnabled: {str(user_settings['voice_enabled']).lower()},
            voiceSpeed: {user_settings['voice_speed']},
            voiceGender: '{user_settings['voice_gender']}',
            voiceLanguage: '{user_settings['voice_language']}',
            mcqCount: {user_settings['mcq_count']}
        }};
        let categoryPerformance = {{}};
        let currentAudio = null;
        let isAudioPlaying = false;
        
        // Test system variables
        let currentTest = null;
        let testTimer = null;
        let testTimeLeft = 30 * 60; // 30 minutes in seconds
        let testStartTime = null;
        let testAnswers = [];
        let currentQuestionIndex = 0;
        let testQuestions = [];
        let testType = ''; // 'session' or 'daily'

        // Goals data
        let goalsData = {{
            total: {total_goals},
            completed: {completed_goals},
            pending: {pending_goals}
        }};

        // Focus sessions data
        let focusSessions = {json.dumps(focus_sessions)};

        // Test sessions data
        let testSessions = {json.dumps(test_sessions)};

        document.addEventListener('DOMContentLoaded', function() {{
            loadHistory();
            document.getElementById('file-input').addEventListener('change', updateFileName);
            document.getElementById('formulas-file-input').addEventListener('change', updateFormulasFileName);
            
            // Apply saved theme
            changeTheme(currentTheme);
            
            // Load daily test status
            updateDailyTestStatus();
        }});

        // Sidebar functionality
        function toggleSidebar() {{
            const sidebar = document.getElementById('sidebar');
            const mainContent = document.getElementById('main-content');
            sidebar.classList.toggle('collapsed');
            mainContent.classList.toggle('expanded');
        }}

        function switchPanel(panelName) {{
            // Update active nav item
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(item => {{
                if (item.onclick && item.onclick.toString().includes(panelName)) {{
                    item.classList.add('active');
                }}
            }});
            
            // Update page title
            const titles = {{
                'generator': 'AI Study Generator',
                'daily-test': 'Daily Revision Test',
                'history': 'Study History',
                'formulas': 'Formulas & Equations',
                'goals': 'Study Goals',
                'focus': 'Focus Mode'
            }};
            document.getElementById('page-title').textContent = titles[panelName] || 'AI Study Suite Pro';
            
            // Switch panel
            document.querySelectorAll('.content-panel').forEach(panel => panel.classList.remove('active'));
            document.getElementById(panelName + '-panel').classList.add('active');
            
            // Load specific data for each panel
            if (panelName === 'history') {{
                loadHistory();
            }} else if (panelName === 'daily-test') {{
                updateDailyTestStatus();
            }} else if (panelName === 'goals') {{
                loadGoals();
            }}
        }}

        function changeTheme(theme) {{
            currentTheme = theme;
            document.body.className = `theme-${{theme}}`;
            
            // Update theme selector
            document.getElementById('theme-selector').value = theme;
            
            // Save theme preference
            fetch('/save_settings', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify({{
                    theme: theme,
                    ai_model: 'gemini-2.5-flash',
                    voice_enabled: userSettings.voiceEnabled,
                    voice_speed: userSettings.voiceSpeed,
                    voice_gender: userSettings.voiceGender,
                    voice_language: userSettings.voiceLanguage,
                    mcq_count: 35
                }})
            }});
        }}

        function showNotification(message, type) {{
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 1rem 1.5rem;
                border-radius: var(--radius);
                color: white;
                font-weight: 600;
                z-index: 10000;
                animation: slideIn 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
                background: ${{type === 'success' ? 'var(--success)' : 'var(--danger)'}};
                box-shadow: var(--shadow-xl);
                backdrop-filter: var(--blur-sm);
                border: 1px solid rgba(255,255,255,0.1);
            `;
            notification.textContent = message;
            document.body.appendChild(notification);
            
            setTimeout(() => {{
                notification.style.animation = 'slideOut 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards';
                setTimeout(() => {{
                    notification.remove();
                }}, 300);
            }}, 3000);
        }}

        // Enhanced Test System Functions
        async function startTestSession(type) {{
            testType = type;
            
            if (type === 'session') {{
                if (!currentContent || !currentContent.mcqs || currentContent.mcqs.length === 0) {{
                    showNotification('No MCQs available for testing. Please generate study materials first.', 'error');
                    return;
                }}
                testQuestions = currentContent.mcqs;
            }} else if (type === 'daily') {{
                try {{
                    const response = await fetch('/generate_daily_test');
                    if (!response.ok) throw new Error('Failed to generate daily test');
                    
                    const testData = await response.json();
                    if (!testData.questions || testData.questions.length === 0) {{
                        showNotification('No questions available for daily test. Please generate some study materials first.', 'error');
                        return;
                    }}
                    testQuestions = testData.questions;
                }} catch (error) {{
                    console.error('Error starting daily test:', error);
                    showNotification('Error starting daily test: ' + error.message, 'error');
                    return;
                }}
            }}
            
            // Initialize test variables
            testTimeLeft = 30 * 60; // 30 minutes
            testStartTime = new Date();
            testAnswers = new Array(testQuestions.length).fill(null);
            currentQuestionIndex = 0;
            
            // Show test interface
            if (type === 'session') {{
                document.getElementById('session-test-interface').style.display = 'block';
                document.getElementById('test-overlay').style.display = 'block';
            }} else {{
                document.getElementById('daily-test-interface').style.display = 'block';
                document.getElementById('start-test-btn').style.display = 'none';
            }}
            
            // Start timer
            startTestTimer();
            
            // Load first question
            loadQuestion(currentQuestionIndex);
            
            showNotification(`${{type === 'session' ? 'Session' : 'Daily'}} test started! You have 30 minutes.`, 'success');
        }}

        function startTestTimer() {{
            if (testTimer) clearInterval(testTimer);
            
            testTimer = setInterval(() => {{
                testTimeLeft--;
                updateTimerDisplay();
                
                if (testTimeLeft <= 0) {{
                    clearInterval(testTimer);
                    submitTest(testType);
                    showNotification('Time is up! Test submitted automatically.', 'warning');
                }}
            }}, 1000);
        }}

        function updateTimerDisplay() {{
            const minutes = Math.floor(testTimeLeft / 60);
            const seconds = testTimeLeft % 60;
            const timerText = `${{minutes.toString().padStart(2, '0')}}:${{seconds.toString().padStart(2, '0')}}`;
            
            if (testType === 'session') {{
                document.getElementById('session-test-timer').textContent = timerText;
            }} else {{
                document.getElementById('daily-test-timer').textContent = timerText;
            }}
        }}

        function loadQuestion(index) {{
            if (index < 0 || index >= testQuestions.length) return;
            
            currentQuestionIndex = index;
            const question = testQuestions[index];
            
            // Update question text
            const questionText = testType === 'session' ? 
                document.getElementById('session-test-question-text') : 
                document.getElementById('daily-test-question-text');
            questionText.textContent = `${{index + 1}}. ${{question.question}}`;
            
            // Update options
            const optionsContainer = testType === 'session' ? 
                document.getElementById('session-test-options') : 
                document.getElementById('daily-test-options');
            
            optionsContainer.innerHTML = question.options.map((option, i) => `
                <div class="option-item ${{testAnswers[index] === String.fromCharCode(65 + i) ? 'selected' : ''}}" 
                     onclick="selectAnswer(${{index}}, ${{i}})">
                    <span style="font-weight: 700; margin-right: 0.5rem;">${{String.fromCharCode(65 + i)}}.</span>
                    ${{option}}
                </div>
            `).join('');
            
            // Update progress
            updateProgress();
            
            // Update navigation buttons
            updateNavigationButtons();
        }}

        function selectAnswer(questionIndex, optionIndex) {{
            testAnswers[questionIndex] = String.fromCharCode(65 + optionIndex);
            loadQuestion(questionIndex); // Reload to update selection
        }}

        function previousQuestion() {{
            if (currentQuestionIndex > 0) {{
                loadQuestion(currentQuestionIndex - 1);
            }}
        }}

        function nextQuestion() {{
            if (currentQuestionIndex < testQuestions.length - 1) {{
                loadQuestion(currentQuestionIndex + 1);
            }}
        }}

        function updateProgress() {{
            const answered = testAnswers.filter(answer => answer !== null).length;
            const progress = (answered / testQuestions.length) * 100;
            const progressText = `${{answered}}/${{testQuestions.length}}`;
            
            if (testType === 'session') {{
                document.getElementById('session-test-progress').style.width = `${{progress}}%`;
                document.getElementById('session-test-progress-text').textContent = progressText;
                document.getElementById('session-test-question-number').textContent = `Question ${{currentQuestionIndex + 1}} of ${{testQuestions.length}}`;
            }} else {{
                document.getElementById('daily-test-progress').style.width = `${{progress}}%`;
                document.getElementById('daily-test-progress-text').textContent = progressText;
                document.getElementById('daily-test-question-number').textContent = `Question ${{currentQuestionIndex + 1}} of ${{testQuestions.length}}`;
            }}
            
            // Show/hide submit button
            const submitBtn = testType === 'session' ? 
                document.getElementById('session-submit-btn') : 
                document.getElementById('daily-submit-btn');
            
            if (answered === testQuestions.length) {{
                submitBtn.style.display = 'block';
            }} else {{
                submitBtn.style.display = 'none';
            }}
        }}

        function updateNavigationButtons() {{
            if (testType === 'session') {{
                document.getElementById('session-prev-btn').disabled = currentQuestionIndex === 0;
                document.getElementById('session-next-btn').disabled = currentQuestionIndex === testQuestions.length - 1;
            }} else {{
                document.getElementById('daily-prev-btn').disabled = currentQuestionIndex === 0;
                document.getElementById('daily-next-btn').disabled = currentQuestionIndex === testQuestions.length - 1;
            }}
        }}

        function submitTest(type) {{
            if (testTimer) {{
                clearInterval(testTimer);
                testTimer = null;
            }}
            
            const timeTaken = 30 * 60 - testTimeLeft;
            const correctAnswers = testQuestions.reduce((count, question, index) => {{
                return count + (testAnswers[index] === question.answer_letter ? 1 : 0);
            }}, 0);
            
            const score = (correctAnswers / testQuestions.length) * 100;
            
            // Calculate performance metrics
            const performance = calculatePerformanceMetrics(testQuestions, testAnswers);
            
            // Show results
            showTestResults(type, score, correctAnswers, testQuestions.length, timeTaken, performance);
            
            // Save test session
            saveTestSession(type, testQuestions, testAnswers, score, timeTaken, performance);
            
            showNotification(`Test submitted! Score: ${{score.toFixed(1)}}%`, 'success');
        }}

        function calculatePerformanceMetrics(questions, answers) {{
            const performance = {{
                categories: {{}},
                difficulties: {{easy: 0, medium: 0, hard: 0}},
                correctByDifficulty: {{easy: 0, medium: 0, hard: 0}},
                totalByDifficulty: {{easy: 0, medium: 0, hard: 0}}
            }};
            
            questions.forEach((question, index) => {{
                const category = question.category || 'General';
                const difficulty = question.difficulty?.toLowerCase() || 'medium';
                const isCorrect = answers[index] === question.answer_letter;
                
                // Track category performance
                if (!performance.categories[category]) {{
                    performance.categories[category] = {{ correct: 0, total: 0 }};
                }}
                performance.categories[category].total++;
                if (isCorrect) {{
                    performance.categories[category].correct++;
                }}
                
                // Track difficulty performance
                performance.difficulties[difficulty]++;
                performance.totalByDifficulty[difficulty]++;
                if (isCorrect) {{
                    performance.correctByDifficulty[difficulty]++;
                }}
            }});
            
            return performance;
        }}

        function showTestResults(type, score, correct, total, timeTaken, performance) {{
            const minutes = Math.floor(timeTaken / 60);
            const seconds = timeTaken % 60;
            const timeText = `${{minutes}}m ${{seconds}}s`;
            
            if (type === 'session') {{
                document.getElementById('session-test-interface').style.display = 'none';
                document.getElementById('session-test-results').style.display = 'block';
                
                // Update results
                document.getElementById('session-circle-score').textContent = `${{score.toFixed(1)}}%`;
                document.getElementById('session-total-questions').textContent = total;
                document.getElementById('session-correct-answers').textContent = correct;
                document.getElementById('session-incorrect-answers').textContent = total - correct;
                document.getElementById('session-time-taken').textContent = timeText;
                
                // Update circle progress
                const circle = document.getElementById('session-score-circle');
                const circumference = 2 * Math.PI * 70;
                const offset = circumference - (score / 100) * circumference;
                circle.style.strokeDasharray = `${{circumference}} ${{circumference}}`;
                circle.style.strokeDashoffset = offset;
                
                // Generate charts
                generateCharts('session', performance);
                generateRecommendations('session', performance, score);
                
            }} else {{
                document.getElementById('daily-test-interface').style.display = 'none';
                document.getElementById('daily-test-results').style.display = 'block';
                
                // Update results
                document.getElementById('daily-circle-score').textContent = `${{score.toFixed(1)}}%`;
                document.getElementById('daily-total-questions').textContent = total;
                document.getElementById('daily-correct-answers').textContent = correct;
                document.getElementById('daily-incorrect-answers').textContent = total - correct;
                document.getElementById('daily-time-taken').textContent = timeText;
                
                // Update circle progress
                const circle = document.getElementById('daily-score-circle');
                const circumference = 2 * Math.PI * 70;
                const offset = circumference - (score / 100) * circumference;
                circle.style.strokeDasharray = `${{circumference}} ${{circumference}}`;
                circle.style.strokeDashoffset = offset;
                
                // Generate charts
                generateCharts('daily', performance);
                generateRecommendations('daily', performance, score);
            }}
        }}

        function generateCharts(prefix, performance) {{
            // Difficulty Chart
            const difficultyCtx = document.getElementById(`${{prefix}}-difficultyChart`).getContext('2d');
            const difficultyData = {{
                labels: ['Easy', 'Medium', 'Hard'],
                datasets: [{{
                    data: [
                        performance.difficulties.easy,
                        performance.difficulties.medium,
                        performance.difficulties.hard
                    ],
                    backgroundColor: [
                        'var(--success)',
                        'var(--warning)',
                        'var(--danger)'
                    ]
                }}]
            }};
            
            new Chart(difficultyCtx, {{
                type: 'pie',
                data: difficultyData,
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'bottom'
                        }}
                    }}
                }}
            }});
            
            // Category Chart
            const categoryCtx = document.getElementById(`${{prefix}}-categoryChart`).getContext('2d');
            const categories = Object.keys(performance.categories);
            const categoryScores = categories.map(cat => {{
                const perf = performance.categories[cat];
                return (perf.correct / perf.total) * 100;
            }});
            
            new Chart(categoryCtx, {{
                type: 'bar',
                data: {{
                    labels: categories,
                    datasets: [{{
                        label: 'Score %',
                        data: categoryScores,
                        backgroundColor: 'var(--primary)',
                        borderColor: 'var(--secondary)',
                        borderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100
                        }}
                    }}
                }}
            }});
        }}

        function generateRecommendations(prefix, performance, score) {{
            const recommendations = [];
            
            // Score-based recommendations
            if (score < 50) {{
                recommendations.push('Focus on understanding basic concepts before attempting advanced questions.');
                recommendations.push('Review the key concepts section thoroughly.');
                recommendations.push('Practice with flashcards to build foundational knowledge.');
            }} else if (score < 70) {{
                recommendations.push('Good progress! Focus on your weaker areas identified below.');
                recommendations.push('Review explanations for incorrect answers carefully.');
                recommendations.push('Consider using the mind map to understand concept relationships.');
            }} else if (score < 90) {{
                recommendations.push('Excellent performance! Refine your knowledge in specific areas.');
                recommendations.push('Pay attention to subtle details in questions.');
                recommendations.push('Challenge yourself with more difficult questions.');
            }} else {{
                recommendations.push('Outstanding performance! Maintain your current study habits.');
                recommendations.push('Consider helping others learn to reinforce your knowledge.');
                recommendations.push('Explore advanced topics beyond the current material.');
            }}
            
            // Performance-based recommendations
            const weakCategories = Object.entries(performance.categories)
                .filter(([cat, perf]) => (perf.correct / perf.total) < 0.7)
                .map(([cat]) => cat);
            
            if (weakCategories.length > 0) {{
                recommendations.push(`Focus on improving these topics: ${{weakCategories.join(', ')}}`);
            }}
            
            // Difficulty-based recommendations
            if (performance.correctByDifficulty.hard / performance.totalByDifficulty.hard < 0.5) {{
                recommendations.push('Practice more hard difficulty questions to build confidence.');
            }}
            
            const container = document.getElementById(`${{prefix}}-recommendations-list`);
            container.innerHTML = recommendations.map(rec => 
                `<div class="recommendation-item">${{rec}}</div>`
            ).join('');
        }}

        function exitTest(type) {{
            if (!confirm('Are you sure you want to exit the test? Your progress will be lost.')) return;
            
            if (testTimer) {{
                clearInterval(testTimer);
                testTimer = null;
            }}
            
            if (type === 'session') {{
                document.getElementById('session-test-interface').style.display = 'none';
                document.getElementById('test-overlay').style.display = 'none';
            }} else {{
                document.getElementById('daily-test-interface').style.display = 'none';
                document.getElementById('start-test-btn').style.display = 'block';
            }}
            
            showNotification('Test exited.', 'info');
        }}

        function retakeTest(type) {{
            if (type === 'session') {{
                document.getElementById('session-test-results').style.display = 'none';
                document.getElementById('test-overlay').style.display = 'none';
            }} else {{
                document.getElementById('daily-test-results').style.display = 'none';
            }}
            startTestSession(type);
        }}

        function closeSessionResults() {{
            document.getElementById('session-test-results').style.display = 'none';
            document.getElementById('test-overlay').style.display = 'none';
        }}

        async function saveTestSession(type, questions, answers, score, timeTaken, performance) {{
            try {{
                const sessionHash = currentContent?.session_hash || 'daily_' + Date.now();
                const response = await fetch('/save_test_session', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        session_hash: sessionHash,
                        test_type: type,
                        questions: questions,
                        user_answers: answers,
                        score: score,
                        time_taken: timeTaken,
                        performance: performance
                    }})
                }});
                
                if (response.ok) {{
                    console.log('Test session saved successfully');
                }}
            }} catch (error) {{
                console.error('Error saving test session:', error);
            }}
        }}

        // Enhanced Chatbot functionality
        function toggleChatbot() {{
            const chatbotWindow = document.getElementById('chatbot-window');
            chatbotWindow.classList.toggle('active');
        }}

        function toggleFormulas() {{
            const formulasSidebar = document.getElementById('formulas-sidebar');
            const chatbotWindow = document.getElementById('chatbot-window');
            
            if (formulasSidebar.classList.contains('active')) {{
                formulasSidebar.classList.remove('active');
                chatbotWindow.style.overflow = 'hidden';
            }} else {{
                formulasSidebar.classList.add('active');
                chatbotWindow.style.overflow = 'hidden';
            }}
        }}

        function handleChatKeypress(event) {{
            if (event.key === 'Enter') {{
                sendMessage();
            }}
        }}

        async function sendMessage() {{
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage('user', message);
            input.value = '';
            
            const typingIndicator = addMessage('bot', 'SRbot is thinking...');
            typingIndicator.classList.add('typing');
            
            try {{
                const response = await fetch('/chat', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        question: message,
                        context: currentDocumentText,
                        formulas: currentFormulas,
                        chat_history: chatHistory
                    }})
                }});
                
                const data = await response.json();
                typingIndicator.remove();
                
                if (data.response) {{
                    const messageElement = addMessage('bot', `<strong>SRbot:</strong> ${{data.response}}`);
                    
                    const actionsDiv = document.createElement('div');
                    actionsDiv.className = 'message-actions';
                    actionsDiv.innerHTML = `
                        <button class="message-action" onclick="speakMessage(this.parentElement.parentElement)">
                            <i class="fas fa-volume-up"></i> Speak
                        </button>
                        <button class="message-action" onclick="stopAudio()" style="display: none;" id="stop-audio-btn">
                            <i class="fas fa-stop"></i> Stop
                        </button>
                    `;
                    messageElement.appendChild(actionsDiv);
                    
                    chatHistory.push({{ role: 'user', content: message }});
                    chatHistory.push({{ role: 'assistant', content: data.response }});
                    
                    if (chatHistory.length > 10) {{
                        chatHistory = chatHistory.slice(-10);
                    }}
                }} else {{
                    addMessage('bot', '<strong>SRbot:</strong> Sorry, I encountered an error. Please try again.');
                }}
            }} catch (error) {{
                typingIndicator.remove();
                addMessage('bot', '<strong>SRbot:</strong> Sorry, I encountered a network or server error. Please check your connection and try again.');
            }}
        }}

        async function speakMessage(messageElement) {{
            if (!userSettings.voiceEnabled) {{
                showNotification('Voice responses are enabled by default. Enjoy!', 'info');
                return;
            }}
            
            if (isAudioPlaying) {{
                stopAudio();
                return;
            }}
            
            const messageText = messageElement.textContent.replace('SRbot:', '').trim();
            const stopButton = messageElement.querySelector('#stop-audio-btn');
            const speakButton = messageElement.querySelector('.message-action[onclick*="speakMessage"]');
            
            try {{
                const response = await fetch('/text_to_speech', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                        }},
                    body: JSON.stringify({{
                        text: messageText,
                        language: userSettings.voiceLanguage,
                        speed: userSettings.voiceSpeed,
                        gender: userSettings.voiceGender
                    }})
                }});
                
                const data = await response.json();
                
                if (data.audio) {{
                    if (currentAudio) {{
                        currentAudio.pause();
                    }}
                    
                    currentAudio = new Audio('data:audio/mp3;base64,' + data.audio);
                    isAudioPlaying = true;
                    
                    if (speakButton) speakButton.style.display = 'none';
                    if (stopButton) stopButton.style.display = 'inline-block';
                    
                    currentAudio.play();
                    
                    currentAudio.onended = function() {{
                        isAudioPlaying = false;
                        if (speakButton) speakButton.style.display = 'inline-block';
                        if (stopButton) stopButton.style.display = 'none';
                    }};
                    
                    currentAudio.onerror = function() {{
                        isAudioPlaying = false;
                        if (speakButton) speakButton.style.display = 'inline-block';
                        if (stopButton) stopButton.style.display = 'none';
                        showNotification('Error playing audio. Please try again.', 'error');
                    }};
                }} else {{
                    showNotification('Failed to generate speech. Check server logs for gTTS errors.', 'error');
                }}
            }} catch (error) {{
                console.error('Error generating speech:', error);
                showNotification('Error generating speech. Please check server logs.', 'error');
            }}
        }}

        function stopAudio() {{
            if (currentAudio) {{
                currentAudio.pause();
                currentAudio.currentTime = 0;
                isAudioPlaying = false;
            }}
            
            // Update all stop/speak buttons
            document.querySelectorAll('#stop-audio-btn').forEach(btn => btn.style.display = 'none');
            document.querySelectorAll('.message-action[onclick*="speakMessage"]').forEach(btn => btn.style.display = 'inline-block');
            
            // Also stop server-side audio if playing
            fetch('/stop_audio', {{ method: 'POST' }});
        }}

        function addMessage(sender, content) {{
            const messagesContainer = document.getElementById('chatbot-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${{sender}}`;
            messageDiv.innerHTML = content;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            return messageDiv;
        }}

        function displayFormulas(formulas) {{
            const formulasList = document.getElementById('formulas-list');
            if (!formulas || formulas.length === 0) {{
                formulasList.innerHTML = '<p style="text-align: center; color: var(--muted); padding: 1.5rem;">No formulas found in the document.</p>';
                return;
            }}
            
            formulasList.innerHTML = formulas.map(formula => 
                `<div class="formula-item">${{formula}}</div>`
            ).join('');
        }}

        function updateFileName() {{
            const fileInput = document.getElementById('file-input');
            const fileName = document.getElementById('file-name');
            if (fileInput.files[0]) {{
                const file = fileInput.files[0];
                const fileSize = (file.size / (1024 * 1024)).toFixed(2);
                fileName.textContent = `${{file.name}} (${{fileSize}} MB)`;
                fileName.style.color = 'var(--success)';
            }} else {{
                fileName.textContent = 'No file selected';
                fileName.style.color = 'var(--muted)';
            }}
        }}

        function updateFormulasFileName() {{
            const fileInput = document.getElementById('formulas-file-input');
            const fileName = document.getElementById('formulas-file-name');
            if (fileInput.files[0]) {{
                const file = fileInput.files[0];
                const fileSize = (file.size / (1024 * 1024)).toFixed(2);
                fileName.textContent = `${{file.name}} (${{fileSize}} MB)`;
                fileName.style.color = 'var(--success)';
            }} else {{
                fileName.textContent = 'No file selected';
                fileName.style.color = 'var(--muted)';
            }}
        }}

        async function generateContent() {{
            const fileInput = document.getElementById('file-input');
            if (!fileInput.files.length) {{
                showNotification('Please select a file first', 'error');
                return;
            }}

            const file = fileInput.files[0];
            const fileSizeMB = file.size / (1024 * 1024);
            
            if (fileSizeMB > 100) {{
                showNotification('File size exceeds 100MB limit. Please choose a smaller file.', 'error');
                return;
            }}

            const generateBtn = document.getElementById('generate-btn');
            const loadingSection = document.getElementById('loading-section');
            const contentSection = document.getElementById('content-section');

            generateBtn.disabled = true;
            generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            loadingSection.style.display = 'block';
            contentSection.style.display = 'none';
            document.getElementById('mcq-analysis-btn').style.display = 'none';

            const formData = new FormData();
            formData.append('notes', file);
            formData.append('mcq_count', 35); // Fixed at 35 MCQs

            try {{
                const response = await fetch('/generate', {{
                    method: 'POST',
                    body: formData
                }});

                if (!response.ok) {{
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Generation failed with status: ' + response.status);
                }}
                
                const responseData = await response.json();
                
                if (responseData.error) {{
                    throw new Error(responseData.error);
                }}

                currentContent = responseData;
                displayContent(currentContent);
                
                const textFormData = new FormData();
                textFormData.append('notes', fileInput.files[0]);

                const textResponse = await fetch('/extract_text', {{
                    method: 'POST',
                    body: textFormData
                }});
                
                if (textResponse.ok) {{
                    const textData = await textResponse.json();
                    currentDocumentText = textData.text || '';
                    currentFormulas = textData.formulas || [];
                    displayFormulas(currentFormulas);
                    
                    const welcomeMsg = document.querySelector('#chatbot-messages .message.bot');
                    if (welcomeMsg) {{
                        welcomeMsg.innerHTML = `<strong>SRbot:</strong> I've analyzed "${{file.name}}". Ask me anything about this document! I found ${{currentFormulas.length}} formulas that might be helpful.`;
                    }}
                }}
                
                await saveToHistory(file.name, currentContent);
                
                contentSection.style.display = 'block';
                showNotification('Study pack generated successfully with 30-40 MCQs!', 'success');

            }} catch (error) {{
                console.error('Generation error:', error);
                showNotification('Error generating content: ' + error.message, 'error');
            }} finally {{
                generateBtn.disabled = false;
                generateBtn.innerHTML = '<i class="fas fa-bolt"></i> Generate Study Pack with SRbot (30-40 MCQs)';
                loadingSection.style.display = 'none';
            }}
        }}

        async function extractFormulas() {{
            const fileInput = document.getElementById('formulas-file-input');
            if (!fileInput.files.length) {{
                showNotification('Please select a file first', 'error');
                return;
            }}

            const file = fileInput.files[0];
            const fileSizeMB = file.size / (1024 * 1024);
            
            if (fileSizeMB > 100) {{
                showNotification('File size exceeds 100MB limit. Please choose a smaller file.', 'error');
                return;
            }}

            const formData = new FormData();
            formData.append('notes', file);

            try {{
                const response = await fetch('/extract_text', {{
                    method: 'POST',
                    body: formData
                }});

                if (!response.ok) {{
                    throw new Error('Extraction failed');
                }}
                
                const data = await response.json();
                currentFormulas = data.formulas || [];
                
                document.getElementById('formulas-content').style.display = 'block';
                displayFormulas(currentFormulas);
                showNotification(`Successfully extracted ${{currentFormulas.length}} formulas`, 'success');

            }} catch (error) {{
                console.error('Formula extraction error:', error);
                showNotification('Error extracting formulas: ' + error.message, 'error');
            }}
        }}

        function displayContent(content) {{
            if (!content) return;
            
            document.getElementById('mcq-performance').style.display = 'none';
            document.getElementById('mcq-analysis-btn').style.display = 'none';
            document.getElementById('mcq-analysis-grid').style.display = 'none';
            
            document.getElementById('concepts-count').textContent = content.key_concepts ? content.key_concepts.length : 0;
            document.getElementById('flashcards-count').textContent = content.flashcards ? content.flashcards.length : 0;
            document.getElementById('mcqs-count').textContent = content.mcqs ? content.mcqs.length : 0;
            
            document.getElementById('concepts-summary').textContent = `${{content.key_concepts ? content.key_concepts.length : 0}} key concepts generated by SRbot`;
            document.getElementById('flashcards-summary').textContent = `${{content.flashcards ? content.flashcards.length : 0}} interactive flashcards`;
            document.getElementById('mcqs-summary').textContent = `${{content.mcqs ? content.mcqs.length : 0}} practice questions (30-40 range)`;
            
            displayConcepts(content.key_concepts || []);
            displayFlashcards(content.flashcards || []);
            displayMCQs(content.mcqs || []);
            displayMindMap(content.mind_map || {{}});
            displayMemoryTricks(content.memory_tricks || {{}});
            
            mcqAnswers = new Array(content.mcqs ? content.mcqs.length : 0).fill(null);
            categoryPerformance = {{}};
            mcqResults = {{
                total: content.mcqs ? content.mcqs.length : 0,
                correct: 0,
                weakAreas: [],
                strongAreas: [],
                detailedAnalysis: {{}}
            }};
            
            // Show test controls for session tests
            document.getElementById('test-controls').style.display = 'block';
        }}

        function displayConcepts(concepts) {{
            const container = document.getElementById('concepts-content');
            if (!concepts || concepts.length === 0) {{
                container.innerHTML = '<div class="text-center"><p>No key concepts generated.</p></div>';
                return;
            }}
            
            container.innerHTML = concepts.map(concept => `
                <div class="concept-card">
                    <span class="concept-importance importance-${{(concept.importance || 'medium').toLowerCase()}}">
                        ${{(concept.importance || 'medium').toUpperCase()}} PRIORITY
                    </span>
                    <h3 style="margin: 1rem 0; color: var(--text); font-size: 1.2rem;">${{concept.concept || 'Unnamed Concept'}}</h3>
                    <p style="color: var(--text); line-height: 1.6; opacity: 0.9; font-size: 1rem;">${{concept.explanation || 'No explanation available.'}}</p>
                </div>
            `).join('');
        }}

        function displayFlashcards(flashcards) {{
            const container = document.getElementById('flashcards-content');
            if (!flashcards || flashcards.length === 0) {{
                container.innerHTML = '<div class="text-center"><p>No flashcards generated.</p></div>';
                return;
            }}
            
            currentFlashcardIndex = 0;
            renderFlashcard(flashcards[currentFlashcardIndex]);
        }}
        
        function renderFlashcard(flashcard) {{
            const container = document.getElementById('flashcards-content');
            container.innerHTML = `
                <div class="flashcard-container">
                    <div class="flashcard" onclick="this.classList.toggle('flipped')">
                        <div class="flashcard-face">
                            <p style="font-size: 1.3rem; font-weight: bold; text-align: center; color: var(--text); margin-bottom: 1rem;">${{flashcard.question || 'No question'}}</p>
                            <small style="color: var(--muted);">Click to flip</small>
                        </div>
                        <div class="flashcard-face flashcard-back">
                            <p style="font-size: 1.1rem; text-align: center; color: var(--text); margin-bottom: 1.2rem;">${{flashcard.answer || 'No answer'}}</p>
                            ${{flashcard.category ? `<div style="padding: 0.6rem; background: var(--primary); color: white; border-radius: var(--radius-sm); font-size: 0.85rem; text-align: center;">Category: ${{flashcard.category}}</div>` : ''}}
                        </div>
                    </div>
                </div>
                <div class="flashcard-nav">
                    <button class="nav-btn" onclick="prevFlashcard()" ${{currentFlashcardIndex === 0 ? 'disabled' : ''}}>
                        <i class="fas fa-chevron-left"></i> Previous
                    </button>
                    <span id="flashcard-counter" style="color: var(--text); font-weight: 600; font-size: 1rem;">
                        ${{currentFlashcardIndex + 1}} / ${{currentContent.flashcards.length}}
                    </span>
                    <button class="nav-btn" onclick="nextFlashcard()" ${{currentFlashcardIndex === currentContent.flashcards.length - 1 ? 'disabled' : ''}}>
                        Next <i class="fas fa-chevron-right"></i>
                    </button>
                </div>
            `;
        }}

        function nextFlashcard() {{
            if (!currentContent?.flashcards?.length) return;
            currentFlashcardIndex = (currentFlashcardIndex + 1) % currentContent.flashcards.length;
            renderFlashcard(currentContent.flashcards[currentFlashcardIndex]);
        }}

        function prevFlashcard() {{
            if (!currentContent?.flashcards?.length) return;
            currentFlashcardIndex = (currentFlashcardIndex - 1 + currentContent.flashcards.length) % currentContent.flashcards.length;
            renderFlashcard(currentContent.flashcards[currentFlashcardIndex]);
        }}

        function displayMCQs(mcqs) {{
            const container = document.getElementById('mcqs-content');
            if (!mcqs || mcqs.length === 0) {{
                container.innerHTML = '<div class="text-center"><p>No MCQs generated.</p></div>';
                return;
            }}
            
            container.innerHTML = mcqs.map((mcq, index) => `
                <div class="mcq-card">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                        <h3 style="color: var(--text); flex: 1; font-size: 1.1rem; line-height: 1.4;">${{index + 1}}. ${{mcq.question || 'No question'}}</h3>
                        <span style="padding: 0.3rem 0.6rem; background: ${{getDifficultyColor(mcq.difficulty)}}; color: white; border-radius: 10px; font-size: 0.75rem; font-weight: 700;">
                            ${{mcq.difficulty || 'Unknown'}}
                        </span>
                    </div>
                    ${{mcq.category ? `<div style="margin-bottom: 1rem; padding: 0.5rem 0.8rem; background: var(--primary); color: white; border-radius: var(--radius-sm); display: inline-block; font-size: 0.85rem; font-weight: 600;">${{mcq.category}}</div>` : ''}}
                    <div class="options-container">
                        ${{(mcq.options || []).map((option, i) => `
                            <div class="option-item" onclick="checkAnswer(this, ${{index}}, '${{String.fromCharCode(65 + i)}}', '${{mcq.answer_letter}}', '${{mcq.category}}', '${{mcq.difficulty}}')">
                                <span style="font-weight: 700; margin-right: 0.6rem;">${{String.fromCharCode(65 + i)}}.</span>
                                ${{option}}
                            </div>
                        `).join('')}}
                    </div>
                    <div class="mcq-explanation">
                        <strong style="color: var(--success); font-size: 1rem;">Correct Answer: ${{mcq.answer_letter}}. ${{mcq.answer_text}}</strong>
                        <p style="margin-top: 1rem; color: var(--text); line-height: 1.5;">${{mcq.explanation || 'No explanation available.'}}</p>
                    </div>
                </div>
            `).join('');
        }}

        function getDifficultyColor(difficulty) {{
            switch(difficulty?.toLowerCase()) {{
                case 'easy': return 'var(--success)';
                case 'medium': return 'var(--warning)';
                case 'hard': return 'var(--danger)';
                default: return 'var(--muted)';
            }}
        }}

        function checkAnswer(selectedOption, questionIndex, selectedLetter, correctLetter, category, difficulty) {{
            if (mcqAnswers[questionIndex] !== null) return;
            
            const mcqCard = selectedOption.closest('.mcq-card');
            const options = mcqCard.querySelectorAll('.option-item');
            const explanation = mcqCard.querySelector('.mcq-explanation');
            
            options.forEach(opt => {{
                opt.style.pointerEvents = 'none';
                opt.onclick = null;
            }});
            
            mcqAnswers[questionIndex] = selectedLetter;
            const isCorrect = selectedLetter === correctLetter;
            
            // Track category performance
            if (category) {{
                if (!categoryPerformance[category]) {{
                    categoryPerformance[category] = {{ correct: 0, total: 0 }};
                }}
                categoryPerformance[category].total++;
                if (isCorrect) {{
                    categoryPerformance[category].correct++;
                    mcqResults.correct++;
                }}
            }} else if (isCorrect) {{
                mcqResults.correct++;
            }}

            // Track difficulty performance
            if (difficulty) {{
                if (!categoryPerformance['difficulty_' + difficulty]) {{
                    categoryPerformance['difficulty_' + difficulty] = {{ correct: 0, total: 0 }};
                }}
                categoryPerformance['difficulty_' + difficulty].total++;
                if (isCorrect) {{
                    categoryPerformance['difficulty_' + difficulty].correct++;
                }}
            }}

            options.forEach(opt => {{
                const optLetter = opt.textContent.trim().charAt(0);
                if (optLetter === correctLetter) {{
                    opt.classList.add('correct');
                }} else if (optLetter === selectedLetter && selectedLetter !== correctLetter) {{
                    opt.classList.add('incorrect');
                }}
            }});
            
            explanation.style.display = 'block';
            checkQuizCompletion();
        }}

        function checkQuizCompletion() {{
            const allAnswered = mcqAnswers.every(answer => answer !== null);
            if (allAnswered) {{
                document.getElementById('mcq-analysis-btn').style.display = 'block';
            }}
        }}

        function showPerformanceReport() {{
            mcqResults.weakAreas = [];
            mcqResults.strongAreas = [];
            
            for (const category in categoryPerformance) {{
                if (category.startsWith('difficulty_')) continue;
                
                const perf = categoryPerformance[category];
                const score = (perf.correct / perf.total);
                
                if (score < 0.6) {{
                    mcqResults.weakAreas.push({{ category, score: Math.round(score * 100), total: perf.total, correct: perf.correct }});
                }}
                if (score >= 0.9) {{
                    mcqResults.strongAreas.push({{ category, score: Math.round(score * 100), total: perf.total, correct: perf.correct }});
                }}
            }}

            mcqResults.weakAreas.sort((a, b) => a.score - b.score);
            mcqResults.strongAreas.sort((a, b) => b.score - a.score);

            const performanceSection = document.getElementById('mcq-performance');
            const scoreValue = document.getElementById('score-value');
            const scoreMessage = document.getElementById('score-message');
            const weakAreasList = document.getElementById('weak-areas-list');
            const strongAreasList = document.getElementById('strong-areas-list');
            
            const percentage = Math.round((mcqResults.correct / mcqResults.total) * 100);
            
            scoreValue.textContent = `${{percentage}}%`;
            
            if (percentage >= 90) {{
                scoreMessage.textContent = 'Excellent! You have mastered this material.';
            }} else if (percentage >= 70) {{
                scoreMessage.textContent = 'Good job! You have a solid understanding.';
            }} else if (percentage >= 50) {{
                scoreMessage.textContent = 'Fair understanding. Review the weak areas.';
            }} else {{
                scoreMessage.textContent = 'Needs improvement. Focus on the weak areas.';
            }}
            
            if (mcqResults.weakAreas.length > 0) {{
                weakAreasList.innerHTML = mcqResults.weakAreas.map(item => 
                    `<li><i class="fas fa-book"></i> ${{item.category}} (${{item.score}}%)</li>`
                ).join('');
            }} else {{
                weakAreasList.innerHTML = '<li>No specific weak areas identified. Great job!</li>';
            }}
            
            if (mcqResults.strongAreas.length > 0) {{
                strongAreasList.innerHTML = mcqResults.strongAreas.map(item => 
                    `<li><i class="fas fa-star"></i> ${{item.category}} (${{item.score}}%)</li>`
                ).join('');
            }} else {{
                strongAreasList.innerHTML = '<li>No specific strong areas identified (below 90% accuracy). Keep practicing!</li>';
            }}
            
            switchTab('mcqs');
            performanceSection.style.display = 'block';
            document.getElementById('mcq-analysis-grid').style.display = 'block';
            
            generateDetailedAnalysis(percentage);
            saveMCQPerformance(percentage, mcqResults.weakAreas.map(a => a.category), mcqResults.strongAreas.map(a => a.category), mcqResults.detailedAnalysis);
        }}

        function generateDetailedAnalysis(percentage) {{
            const overallPerformance = document.getElementById('overall-performance');
            const mistakesAnalysis = document.getElementById('mistakes-analysis');
            const improvementSuggestions = document.getElementById('improvement-suggestions');
            const strengthsAnalysis = document.getElementById('strengths-analysis');
            
            const incorrectCount = mcqResults.total - mcqResults.correct;
            overallPerformance.innerHTML = `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 0.8rem;">
                    <div style="text-align: center; padding: 0.8rem; background: var(--glass-dark); border-radius: var(--radius-sm);">
                        <div style="font-size: 1.3rem; font-weight: 800; color: var(--primary);">${{mcqResults.total}}</div>
                        <div style="color: var(--text); font-size: 0.85rem;">Total Questions</div>
                    </div>
                    <div style="text-align: center; padding: 0.8rem; background: var(--glass-dark); border-radius: var(--radius-sm);">
                        <div style="font-size: 1.3rem; font-weight: 800; color: var(--success);">${{mcqResults.correct}}</div>
                        <div style="color: var(--text); font-size: 0.85rem;">Correct Answers</div>
                    </div>
                    <div style="text-align: center; padding: 0.8rem; background: var(--glass-dark); border-radius: var(--radius-sm);">
                        <div style="font-size: 1.3rem; font-weight: 800; color: var(--danger);">${{incorrectCount}}</div>
                        <div style="color: var(--text); font-size: 0.85rem;">Incorrect Answers</div>
                    </div>
                    <div style="text-align: center; padding: 0.8rem; background: var(--glass-dark); border-radius: var(--radius-sm);">
                        <div style="font-size: 1.3rem; font-weight: 800; color: var(--accent);">${{percentage}}%</div>
                        <div style="color: var(--text); font-size: 0.85rem;">Success Rate</div>
                    </div>
                </div>
            `;
            
            mistakesAnalysis.innerHTML = `
                <p style="margin-bottom: 0.8rem;">You answered <strong style="color: var(--danger);">${{incorrectCount}}</strong> questions incorrectly.</p>
                ${{mcqResults.weakAreas.length > 0 ? `
                    <p style="font-weight: 600; margin-bottom: 0.6rem;"><strong>Topics needing immediate review:</strong></p>
                    <ul style="list-style: none; padding: 0;">
                        ${{mcqResults.weakAreas.map(item => `
                            <li style="padding: 0.6rem; margin-bottom: 0.4rem; background: var(--glass-dark); border-radius: var(--radius-sm); border-left: 3px solid var(--warning);">
                                <i class="fas fa-exclamation-circle" style="color: var(--warning); margin-right: 0.4rem;"></i>
                                ${{item.category}} (Score: <strong style="color: var(--warning);">${{item.score}}%</strong>)
                            </li>
                        `).join('')}}
                    </ul>
                ` : '<p style="color: var(--success); font-weight: 600;">Great job! You showed a high level of accuracy across all tested categories.</p>'}}
            `;
            
            strengthsAnalysis.innerHTML = `
                ${{mcqResults.strongAreas.length > 0 ? `
                    <p style="font-weight: 600; margin-bottom: 0.6rem;"><strong>Your strong areas (90%+ Accuracy):</strong></p>
                    <ul style="list-style: none; padding: 0;">
                        ${{mcqResults.strongAreas.map(item => `
                            <li style="padding: 0.6rem; margin-bottom: 0.4rem; background: var(--glass-dark); border-radius: var(--radius-sm); border-left: 3px solid var(--success);">
                                <i class="fas fa-star" style="color: var(--success); margin-right: 0.4rem;"></i>
                                ${{item.category}} (Score: <strong style="color: var(--success);">${{item.score}}%</strong>)
                            </li>
                        `).join('')}}
                    </ul>
                ` : '<p style="color: var(--warning);">No specific strong areas identified (below 90% accuracy). Keep practicing!</p>'}}
            `;

            let suggestions = [];
            if (percentage < 50) {{
                suggestions = [
                    "Focus on understanding the basic concepts before attempting advanced questions. Start with the Key Concepts section.",
                    "Spend more time on the lowest scoring weak areas identified above.",
                    "Practice with flashcards to build foundational knowledge.",
                    "Consider studying in smaller, more frequent sessions to improve retention."
                ];
            }} else if (percentage < 70) {{
                suggestions = [
                    "Review the explanations for your incorrect answers carefully to close knowledge gaps.",
                    "Focus on the weak areas identified above, perhaps by chatting with SRbot about those specific topics.",
                    "Use the mind map to better understand the relationships between concepts and prevent isolated knowledge.",
                    "Try the memory tricks to improve retention of key facts and formulas."
                ];
            }} else if (percentage < 90) {{
                suggestions = [
                    "You have a good understanding—focus on refining your knowledge of the weak areas (60-89% score).",
                    "Pay attention to the subtle details in questions, particularly the 'hard' difficulty ones.",
                    "Review the formulas and complex concepts to achieve 100% mastery.",
                    "Continue regular review of your strong areas to maintain your high level of understanding."
                ];
            }} else {{
                suggestions = [
                    "Excellent performance! Maintain your current study habits and consider challenging yourself with more advanced materials.",
                    "Help others learn—teaching reinforces your own knowledge deeply.",
                    "Review the few questions you missed to ensure complete mastery.",
                    "Utilize the SRbot Chat Assistant to explore adjacent or related topics not covered in the document."
                ];
            }}
            
            improvementSuggestions.innerHTML = `
                <ul style="list-style: none; padding: 0;">
                    ${{suggestions.map(suggestion => `
                        <li style="padding: 0.8rem; margin-bottom: 0.6rem; background: var(--glass-dark); border-radius: var(--radius-sm); border-left: 3px solid var(--info);">
                            <i class="fas fa-check-circle" style="color: var(--info); margin-right: 0.4rem;"></i>
                            ${{suggestion}}
                        </li>
                    `).join('')}}
                </ul>
            `;
            
            mcqResults.detailedAnalysis = {{
                overallPerformance: {{
                    total: mcqResults.total,
                    correct: mcqResults.correct,
                    incorrect: incorrectCount,
                    percentage: percentage
                }},
                weakAreas: mcqResults.weakAreas,
                strongAreas: mcqResults.strongAreas,
                suggestions: suggestions
            }};
        }}

        function toggleDetailedAnalysis() {{
            const analysisSection = document.getElementById('detailed-analysis');
            const button = document.querySelector('.detailed-analysis-btn');
            
            if (analysisSection.style.display === 'none' || !analysisSection.style.display) {{
                analysisSection.style.display = 'block';
                button.innerHTML = '<i class="fas fa-chart-bar"></i> Hide Detailed Analysis';
            }} else {{
                analysisSection.style.display = 'none';
                button.innerHTML = '<i class="fas fa-chart-bar"></i> Show Detailed Analysis';
            }}
        }}

        function resetMCQs() {{
            mcqAnswers = new Array(currentContent.mcqs.length).fill(null);
            categoryPerformance = {{}};
            mcqResults = {{
                total: currentContent.mcqs.length,
                correct: 0,
                weakAreas: [],
                strongAreas: [],
                detailedAnalysis: {{}}
            }};
            
            document.getElementById('mcq-performance').style.display = 'none';
            document.getElementById('detailed-analysis').style.display = 'none';
            document.getElementById('mcq-analysis-grid').style.display = 'none';
            document.getElementById('mcq-analysis-btn').style.display = 'none';
            document.querySelector('.detailed-analysis-btn').innerHTML = '<i class="fas fa-chart-bar"></i> Show Detailed Analysis';
            displayMCQs(currentContent.mcqs);
        }}
        
        function displayMindMap(mindMap) {{
            const container = document.getElementById('mindmap-content');
            if (!mindMap || Object.keys(mindMap).length === 0) {{
                container.innerHTML = '<div class="text-center"><p>No mind map generated.</p></div>';
                return;
            }}

            container.innerHTML = `
                <div class="mind-map-visual">
                    <div class="central-topic">${{mindMap.central_topic || 'Central Topic'}}</div>
                    
                    ${{mindMap.main_branches && mindMap.main_branches.length > 0 ? `
                        <div class="main-branches">
                            ${{mindMap.main_branches.map(branch => `
                                <div class="branch">
                                    <div class="branch-title">${{branch.name || 'Branch'}}</div>
                                    ${{branch.sub_branches && branch.sub_branches.length > 0 ? `
                                        <ul class="sub-branches">
                                            ${{branch.sub_branches.map(sub => `<li>${{sub}}</li>`).join('')}}
                                        </ul>
                                    ` : '<p style="color: var(--muted); text-align: center;">No sub-branches</p>'}}
                                </div>
                            `).join('')}}
                        </div>
                    ` : '<p style="color: var(--muted); text-align: center;">No main branches generated</p>'}}
                    
                    ${{mindMap.connections ? `
                        <div style="margin-top: 1.5rem; padding: 1.5rem; background: var(--glass-dark); border-radius: var(--radius); border-left: 4px solid var(--accent); backdrop-filter: var(--blur-sm);">
                            <strong style="color: var(--accent); font-size: 1.1rem;">Key Connections:</strong>
                            <p style="margin-top: 0.8rem; color: var(--text); line-height: 1.5;">${{mindMap.connections}}</p>
                        </div>
                    ` : ''}}
                </div>
            `;
        }}

        function displayMemoryTricks(tricks) {{
            const container = document.getElementById('tricks-content');
            if (!tricks || Object.keys(tricks).length === 0) {{
                container.innerHTML = '<div class="text-center"><p>No memory tricks generated.</p></div>';
                return;
            }}

            container.innerHTML = `
                <div class="trick-card">
                    <div class="trick-icon"><i class="fas fa-abacus"></i></div>
                    <h3>Acronyms & Mnemonics</h3>
                    <p style="color: var(--text); line-height: 1.5; font-size: 1rem;">${{tricks.acronyms || 'No acronyms generated. Try using the first letters of key terms to create memorable words or phrases.'}}</p>
                </div>
                <div class="trick-card">
                    <div class="trick-icon"><i class="fas fa-music"></i></div>
                    <h3>Rhymes & Songs</h3>
                    <p style="color: var(--text); line-height: 1.5; font-size: 1rem;">${{tricks.rhymes || 'No rhymes generated. Create simple rhymes or set information to familiar tunes to improve recall.'}}</p>
                </div>
                <div class="trick-card">
                    <div class="trick-icon"><i class="fas fa-eye"></i></div>
                    <h3>Visual Associations</h3>
                    <p style="color: var(--text); line-height: 1.5; font-size: 1rem;">${{tricks.visual_associations || 'No visual associations generated. Connect concepts with vivid mental images to enhance memory.'}}</p>
                </div>
                <div class="trick-card">
                    <div class="trick-icon"><i class="fas fa-book"></i></div>
                    <h3>Story Method</h3>
                    <p style="color: var(--text); line-height: 1.5; font-size: 1rem;">${{tricks.story_method || 'No story method generated. Create a narrative that links concepts together in a meaningful sequence.'}}</p>
                </div>
            `;
        }}

        function switchTab(tabName) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelector(`.tab-btn[onclick="switchTab('${{tabName}}')"]`).classList.add('active');
            
            document.querySelectorAll('#content-section .content-panel').forEach(panel => panel.classList.remove('active'));
            document.getElementById(tabName + '-panel').classList.add('active');
        }}

        async function saveToHistory(filename, content) {{
            try {{
                const contentToSend = {{...content}};
                delete contentToSend.full_content;

                const response = await fetch('/save_history', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ filename, content: contentToSend }})
                }});
                
                if (!response.ok) throw new Error('Failed to save history');
                const data = await response.json();
                currentContent.session_hash = data.session_hash;
                console.log(`History saved successfully with hash: ${{data.session_hash}}`);
            }} catch (error) {{
                console.error("Failed to save history:", error);
            }}
        }}

        async function saveMCQPerformance(score, weakAreas, strongAreas, detailedAnalysis) {{
            if (!currentContent.session_hash) return;

            try {{
                const response = await fetch('/save_mcq_performance', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        session_hash: currentContent.session_hash,
                        total_questions: mcqResults.total,
                        correct_answers: mcqResults.correct,
                        weak_areas: weakAreas,
                        strong_areas: strongAreas,
                        detailed_analysis: detailedAnalysis
                    }})
                }});
                
                if (!response.ok) throw new Error('Failed to save performance');
                console.log('MCQ performance saved successfully');
            }} catch (error) {{
                console.error("Failed to save MCQ performance:", error);
            }}
        }}

        async function loadHistory() {{
            try {{
                const response = await fetch('/get_history');
                if (!response.ok) throw new Error('Failed to load history');
                
                const history = await response.json();
                const container = document.getElementById('history-content');
                
                if (!history || history.length === 0) {{
                    container.innerHTML = '<div class="text-center"><p>No study history yet. Generate your first study pack with SRbot to see it here!</p></div>';
                    return;
                }}
                
                container.innerHTML = history.map(item => `
                    <div class="history-item" onclick="loadHistoryItem('${{item.session_hash}}')">
                        <h4 style="margin-bottom: 0.8rem; color: var(--text); font-size: 1.1rem;">${{item.filename}}</h4>
                        <p style="color: var(--text); opacity: 0.9; margin-bottom: 0.8rem; line-height: 1.4;">${{item.content_preview}}...</p>
                        <div class="history-stats">
                            <span class="history-stat">Concepts: ${{item.key_concepts_count}}</span>
                            <span class="history-stat">Flashcards: ${{item.flashcards_count}}</span>
                            <span class="history-stat">MCQs: ${{item.mcqs_count}}</span>
                        </div>
                        <small style="color: var(--muted); margin-top: 0.8rem; display: block;">Created: ${{new Date(item.created_at).toLocaleString()}}</small>
                    </div>
                `).join('');
            }} catch (error) {{
                console.error("Failed to load history:", error);
                document.getElementById('history-content').innerHTML = '<div class="text-center"><p>Failed to load study history.</p></div>';
            }}
        }}

        async function loadHistoryItem(sessionHash) {{
            try {{
                const response = await fetch(`/get_session/${{sessionHash}}`);
                if (!response.ok) throw new Error('Session not found');
                
                const content = await response.json();
                currentContent = content;
                
                switchPanel('generator');
                document.getElementById('content-section').style.display = 'block';
                displayContent(content);
                window.scrollTo(0, 0);
                showNotification('Study session loaded successfully!', 'success');
            }} catch (error) {{
                console.error("Failed to load history item:", error);
                showNotification('Failed to load study session. It may have been deleted.', 'error');
            }}
        }}

        // Daily Test Functions
        async function updateDailyTestStatus() {{
            try {{
                const response = await fetch('/get_daily_test');
                if (response.ok) {{
                    const testData = await response.json();
                    if (testData) {{
                        document.getElementById('test-questions-count').textContent = testData.questions ? testData.questions.length : 0;
                        document.getElementById('test-completed').textContent = testData.completed ? 'Yes' : 'No';
                        document.getElementById('test-score').textContent = testData.completed ? `${{testData.score}}%` : 'N/A';
                        
                        if (testData.completed) {{
                            document.getElementById('start-test-btn').innerHTML = '<i class="fas fa-redo"></i> Retry Test';
                        }}
                    }}
                }}
            }} catch (error) {{
                console.error('Error updating daily test status:', error);
            }}
        }}

        // Goals/To-do List Functions
        async function addGoal() {{
            const title = document.getElementById('goal-title').value.trim();
            const description = document.getElementById('goal-description').value.trim();
            const category = document.getElementById('goal-category').value;
            const priority = document.getElementById('goal-priority').value;
            const dueDate = document.getElementById('goal-due-date').value;
            
            if (!title) {{
                showNotification('Please enter a goal title', 'error');
                return;
            }}
            
            try {{
                const response = await fetch('/add_goal', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        title: title,
                        description: description,
                        category: category,
                        priority: priority,
                        due_date: dueDate
                    }})
                }});
                
                if (response.ok) {{
                    // Clear form
                    document.getElementById('goal-title').value = '';
                    document.getElementById('goal-description').value = '';
                    document.getElementById('goal-due-date').value = '';
                    
                    // Reload goals
                    loadGoals();
                    showNotification('Goal added successfully!', 'success');
                }} else {{
                    throw new Error('Failed to add goal');
                }}
            }} catch (error) {{
                console.error('Error adding goal:', error);
                showNotification('Error adding goal: ' + error.message, 'error');
            }}
        }}

        async function toggleGoal(goalId, checkbox) {{
            const completed = checkbox.innerHTML.includes('fa-check');
            
            try {{
                const response = await fetch('/update_goal_status', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        goal_id: goalId,
                        completed: !completed
                    }})
                }});
                
                if (response.ok) {{
                    loadGoals();
                }} else {{
                    throw new Error('Failed to update goal');
                }}
            }} catch (error) {{
                console.error('Error updating goal:', error);
                showNotification('Error updating goal: ' + error.message, 'error');
            }}
        }}

        async function deleteGoal(goalId) {{
            if (!confirm('Are you sure you want to delete this goal?')) return;
            
            try {{
                const response = await fetch('/delete_goal', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        goal_id: goalId
                    }})
                }});
                
                if (response.ok) {{
                    loadGoals();
                    showNotification('Goal deleted successfully!', 'success');
                }} else {{
                    throw new Error('Failed to delete goal');
                }}
            }} catch (error) {{
                console.error('Error deleting goal:', error);
                showNotification('Error deleting goal: ' + error.message, 'error');
            }}
        }}

        async function loadGoals() {{
            try {{
                const response = await fetch('/get_goals');
                if (response.ok) {{
                    const goals = await response.json();
                    const container = document.getElementById('goals-list');
                    
                    if (!goals || goals.length === 0) {{
                        container.innerHTML = `
                            <div class="text-center" style="padding: 2rem; color: var(--muted);">
                                <i class="fas fa-bullseye" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                                <p>No goals set yet. Create your first study goal above!</p>
                            </div>
                        `;
                        return;
                    }}
                    
                    container.innerHTML = goals.map(goal => `
                        <div class="goal-item ${{goal.completed ? 'completed' : ''}}" data-id="${{goal.id}}">
                            <div class="goal-checkbox" onclick="toggleGoal(${{goal.id}}, this)">
                                ${{goal.completed ? '<i class="fas fa-check" style="color: var(--success);"></i>' : ''}}
                            </div>
                            <div class="goal-content">
                                <div class="goal-title">${{goal.title}}</div>
                                <div class="goal-description">${{goal.description}}</div>
                                <div class="goal-meta">
                                    <span class="goal-priority priority-${{goal.priority}}">${{goal.priority.toUpperCase()}}</span>
                                    <span>${{goal.category}}</span>
                                    <span>${{goal.due_date || 'No due date'}}</span>
                                </div>
                            </div>
                            <div class="goal-actions">
                                <button class="goal-action" onclick="deleteGoal(${{goal.id}})">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        </div>
                    `).join('');
                    
                    // Update stats
                    const totalGoals = goals.length;
                    const completedGoals = goals.filter(g => g.completed).length;
                    const pendingGoals = totalGoals - completedGoals;
                    const completionRate = totalGoals > 0 ? Math.round((completedGoals / totalGoals) * 100) : 0;
                    
                    document.getElementById('total-goals').textContent = totalGoals;
                    document.getElementById('completed-goals').textContent = completedGoals;
                    document.getElementById('pending-goals').textContent = pendingGoals;
                    document.getElementById('completion-rate').textContent = completionRate + '%';
                }}
            }} catch (error) {{
                console.error('Error loading goals:', error);
            }}
        }}

        // Focus Mode Functions
        function startFocusTimer() {{
            if (isFocusRunning) return;
            
            isFocusRunning = true;
            focusTimer = setInterval(() => {{
                focusTimeLeft--;
                updateFocusTimerDisplay();
                
                if (focusTimeLeft <= 0) {{
                    clearInterval(focusTimer);
                    isFocusRunning = false;
                    showNotification('Focus session completed! Time for a break.', 'success');
                    saveFocusSession(25, 'pomodoro', 'Completed focus session');
                }}
            }}, 1000);
        }}

        function pauseFocusTimer() {{
            if (!isFocusRunning) return;
            
            clearInterval(focusTimer);
            isFocusRunning = false;
        }}

        function resetFocusTimer() {{
            clearInterval(focusTimer);
            isFocusRunning = false;
            focusTimeLeft = 25 * 60;
            updateFocusTimerDisplay();
        }}

        function updateFocusTimerDisplay() {{
            const minutes = Math.floor(focusTimeLeft / 60);
            const seconds = focusTimeLeft % 60;
            document.getElementById('focus-timer').textContent = 
                `${{minutes.toString().padStart(2, '0')}}:${{seconds.toString().padStart(2, '0')}}`;
        }}

        function enterFocusMode() {{
            isFocusModeActive = true;
            document.body.classList.add('focus-mode-active');
            document.getElementById('focus-exit-btn').style.display = 'block';
            showNotification('Focus mode activated. Minimize distractions and focus!', 'success');
        }}

        function exitFocusMode() {{
            isFocusModeActive = false;
            document.body.classList.remove('focus-mode-active');
            document.getElementById('focus-exit-btn').style.display = 'none';
            showNotification('Focus mode deactivated.', 'info');
        }}

        async function saveFocusSession(duration, sessionType, notes) {{
            try {{
                const response = await fetch('/save_focus_session', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        duration: duration,
                        session_type: sessionType,
                        notes: notes
                    }})
                }});
                
                if (response.ok) {{
                    console.log('Focus session saved');
                }}
            }} catch (error) {{
                console.error('Error saving focus session:', error);
            }}
        }}

        // Add CSS animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {{
                from {{ transform: translateX(100%); opacity: 0; }}
                to {{ transform: translateX(0); opacity: 1; }}
            }}
            
            @keyframes slideOut {{
                from {{ transform: translateX(0); opacity: 1; }}
                to {{ transform: translateX(100%); opacity: 0; }}
            }}
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>
    """
    return render_template_string(html_content)

# Goals/To-do list routes
@app.route('/get_goals')
def get_goals_route():
    """Get user goals."""
    try:
        goals = get_user_goals()
        return jsonify(goals)
    except Exception as e:
        print(f"❌ Error getting goals: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/add_goal', methods=['POST'])
def add_goal_route():
    """Add a new goal."""
    try:
        data = request.json
        title = data.get('title', '')
        description = data.get('description', '')
        category = data.get('category', 'study')
        priority = data.get('priority', 'medium')
        due_date = data.get('due_date', '')
        
        if not title:
            return jsonify({"error": "Goal title is required"}), 400
        
        success = save_user_goal(title, description, category, priority, due_date)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to save goal"}), 500
            
    except Exception as e:
        print(f"❌ Error adding goal: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/update_goal_status', methods=['POST'])
def update_goal_status_route():
    """Update goal completion status."""
    try:
        data = request.json
        goal_id = data.get('goal_id')
        completed = data.get('completed', False)
        
        if not goal_id:
            return jsonify({"error": "Goal ID is required"}), 400
        
        success = update_goal_status(goal_id, completed)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to update goal"}), 500
            
    except Exception as e:
        print(f"❌ Error updating goal: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/delete_goal', methods=['POST'])
def delete_goal_route():
    """Delete a goal."""
    try:
        data = request.json
        goal_id = data.get('goal_id')
        
        if not goal_id:
            return jsonify({"error": "Goal ID is required"}), 400
        
        success = delete_goal(goal_id)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to delete goal"}), 500
            
    except Exception as e:
        print(f"❌ Error deleting goal: {e}")
        return jsonify({"error": str(e)}), 500

# Focus mode routes
@app.route('/save_focus_session', methods=['POST'])
def save_focus_session_route():
    """Save focus session."""
    try:
        data = request.json
        duration = data.get('duration', 0)
        session_type = data.get('session_type', 'pomodoro')
        notes = data.get('notes', '')
        
        success = save_focus_session(duration, session_type, notes)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to save focus session"}), 500
            
    except Exception as e:
        print(f"❌ Error saving focus session: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_focus_sessions')
def get_focus_sessions_route():
    """Get focus sessions."""
    try:
        days = request.args.get('days', 30, type=int)
        sessions = get_focus_sessions(days)
        return jsonify(sessions)
    except Exception as e:
        print(f"❌ Error getting focus sessions: {e}")
        return jsonify({"error": str(e)}), 500

# Enhanced Test Session routes
@app.route('/save_test_session', methods=['POST'])
def save_test_session_route():
    """Save test session."""
    try:
        data = request.json
        session_hash = data.get('session_hash', '')
        test_type = data.get('test_type', 'session')
        questions = data.get('questions', [])
        user_answers = data.get('user_answers', [])
        score = data.get('score', 0)
        time_taken = data.get('time_taken', 0)
        performance = data.get('performance', {})
        
        success = save_test_session(session_hash, test_type, questions, user_answers, score, time_taken)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to save test session"}), 500
            
    except Exception as e:
        print(f"❌ Error saving test session: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_test_sessions')
def get_test_sessions_route():
    """Get test sessions."""
    try:
        days = request.args.get('days', 30, type=int)
        sessions = get_test_sessions(days)
        return jsonify(sessions)
    except Exception as e:
        print(f"❌ Error getting test sessions: {e}")
        return jsonify({"error": str(e)}), 500

# Original routes
@app.route('/generate_daily_test')
def generate_daily_test_route():
    """Generate daily revision test."""
    questions = generate_daily_revision_test()
    if questions:
        return jsonify({"questions": questions, "count": len(questions)})
    else:
        return jsonify({"error": "No study sessions found to generate test from"}), 404

@app.route('/get_daily_test')
def get_daily_test_route():
    """Get today's revision test."""
    test_data = get_daily_revision_test()
    if test_data:
        return jsonify(test_data)
    else:
        return jsonify({"error": "No test found for today"}), 404

@app.route('/save_daily_test_result', methods=['POST'])
def save_daily_test_result_route():
    """Save daily test results."""
    try:
        data = request.json
        score = data.get('score', 0)
        analysis_data = data.get('analysis_data', {})
        
        success = save_daily_revision_result(score, analysis_data)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to save test results"}), 500
            
    except Exception as e:
        print(f"❌ Error saving daily test result: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def handle_generation():
    if 'notes' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['notes']
    mcq_count = request.form.get('mcq_count', 35, type=int)
    
    # Fixed MCQ count between 30-40
    mcq_count = max(30, min(40, mcq_count))
    
    file_bytes = file.read()
    file.seek(0)
    file_io = io.BytesIO(file_bytes)
    file_io.filename = file.filename
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    file_type = file.filename.split('.')[-1].lower()
    
    try:
        if file_type == 'pdf':
            notes_text = extract_text_from_pdf(file_io) 
        elif file_type == 'txt':
            notes_text = file_bytes.decode('utf-8')
        else:
            return jsonify({"error": "Unsupported file type. Please upload a PDF or TXT file."}), 400
        
        if not notes_text.strip():
            return jsonify({"error": "Could not extract text from file or file is empty."}), 400
        
        print(f"📝 Processing file with {len(notes_text)} characters, generating {mcq_count} MCQs")
        
        all_content = generate_all_content(notes_text, mcq_count)
        
        if "error" in all_content:
            return jsonify(all_content), 500
            
        return jsonify(all_content)
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return jsonify({"error": f"An error occurred while processing the file: {str(e)}"}), 500

@app.route('/extract_text', methods=['POST'])
def extract_text():
    """Extract text and formulas from uploaded file for chatbot."""
    if 'notes' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['notes']
    file_bytes = file.read()
    file.seek(0)
    file_io = io.BytesIO(file_bytes)
    file_io.filename = file.filename

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    file_type = file.filename.split('.')[-1].lower()
    
    try:
        if file_type == 'pdf':
            text = extract_text_from_pdf(file_io)
        elif file_type == 'txt':
            text = file_bytes.decode('utf-8')
        else:
            return jsonify({"error": "Unsupported file type"}), 400
        
        formulas = extract_formulas_from_text(text)
        
        return jsonify({
            "text": text,
            "formulas": formulas,
            "filename": file.filename
        })
        
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def handle_chat():
    """Handle chat messages with context from uploaded document."""
    try:
        data = request.json
        question = data.get('question', '')
        context = data.get('context', '')
        formulas = data.get('formulas', [])
        chat_history = data.get('chat_history', [])
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        response = chat_with_ai(question, context, formulas, chat_history)
        
        return jsonify({"response": response})
        
    except Exception as e:
        print(f"❌ Error in chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/text_to_speech', methods=['POST'])
def handle_text_to_speech():
    """Convert text to speech with customizable voice settings."""
    try:
        data = request.json
        text = data.get('text', '')
        language = data.get('language', 'en')
        speed = data.get('speed', 1.0)
        gender = data.get('gender', 'female')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        text = text[:1000]
        
        audio_base64 = text_to_speech(text, language, speed, gender)
        
        if audio_base64:
            return jsonify({"audio": audio_base64})
        else:
            return jsonify({"error": "Failed to generate speech"}), 500
            
    except Exception as e:
        print(f"❌ Error in text-to-speech: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stop_audio', methods=['POST'])
def handle_stop_audio():
    """Stop currently playing audio."""
    try:
        success = stop_audio()
        return jsonify({"success": success})
    except Exception as e:
        print(f"❌ Error stopping audio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save_history', methods=['POST'])
def save_history():
    try:
        data = request.json
        filename = data.get('filename', 'Unknown File')
        content = data.get('content', {})
        
        content_preview = f"Generated study materials with {len(content.get('key_concepts', []))} concepts"
        
        session_hash = save_study_session(filename, content_preview, content)
        
        if session_hash:
            return jsonify({"success": True, "session_hash": session_hash})
        else:
            return jsonify({"error": "Failed to save session"}), 500
            
    except Exception as e:
        print(f"❌ Error in /save_history route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save_mcq_performance', methods=['POST'])
def save_mcq_performance_route():
    try:
        data = request.json
        session_hash = data.get('session_hash', '')
        total_questions = data.get('total_questions', 0)
        correct_answers = data.get('correct_answers', 0)
        weak_areas = data.get('weak_areas', [])
        strong_areas = data.get('strong_areas', [])
        detailed_analysis = data.get('detailed_analysis', {})
        
        success = save_mcq_performance(session_hash, total_questions, correct_answers, weak_areas, strong_areas, detailed_analysis)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to save MCQ performance"}), 500
            
    except Exception as e:
        print(f"❌ Error in /save_mcq_performance route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save_settings', methods=['POST'])
def save_settings():
    try:
        data = request.json
        theme = data.get('theme', 'dark')
        ai_model = data.get('ai_model', 'gemini-2.5-flash')
        voice_enabled = data.get('voice_enabled', True)
        voice_speed = data.get('voice_speed', 1.0)
        voice_gender = data.get('voice_gender', 'female')
        voice_language = data.get('voice_language', 'en')
        mcq_count = data.get('mcq_count', 35)
        
        success = save_user_settings(theme, ai_model, voice_enabled, voice_speed, voice_gender, voice_language, mcq_count)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to save settings"}), 500
            
    except Exception as e:
        print(f"❌ Error in /save_settings route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_history')
def get_history():
    try:
        history = get_study_history()
        return jsonify(history)
    except Exception as e:
        print(f"❌ Error in /get_history route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_session/<session_hash>')
def get_session(session_hash):
    try:
        session_data = get_study_session_by_hash(session_hash)
        if session_data:
            return jsonify(session_data)
        else:
            return jsonify({"error": "Session not found"}), 404
    except Exception as e:
        print(f"❌ Error in /get_session route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_test_scores')
def get_test_scores_route():
    """Get test scores for analytics."""
    try:
        days = request.args.get('days', 30, type=int)
        scores = get_test_scores(days)
        return jsonify(scores)
    except Exception as e:
        print(f"❌ Error getting test scores: {e}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 100MB."}), 413

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

if __name__ == '__main__':
    print("🚀 Starting AI Study Suite Pro Server...")
    print("📚 Premium features enabled:")
    print("   - AI-Powered Study Pack Generation")
    print("   - Interactive Flashcards & MCQs")
    print("   - Enhanced Test System with Timer")
    print("   - Daily Revision Tests with AI Analysis")
    print("   - Study History & Performance Tracking")
    print("   - Goals/To-do List Management")
    print("   - Focus Mode with Timer")
    print("   - Voice Assistance with SRbot")
    print("   - Multiple Theme Options")
    print("   - Formula Extraction & Chat Integration")
    
    # Check if Gemini API is configured
    if model:
        print("✅ Gemini AI: Configured and Ready")
    else:
        print("⚠️  Gemini AI: Not configured - AI features disabled")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)