# app.py

import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
from keras.models import model_from_json
import operator
import difflib
from string import ascii_uppercase
import threading
import time
import config

# Optional: enchant dictionary for better suggestions
try:
    import enchant
    ENCHANT_AVAILABLE = True
except Exception:
    ENCHANT_AVAILABLE = False

# Optional: wordfreq for a large high-quality frequency list
try:
    from wordfreq import top_n_list
    WORDFREQ_AVAILABLE = True
except Exception:
    WORDFREQ_AVAILABLE = False

# Fallback words if nothing else exists
FALLBACK_WORDS = [
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on', 'with', 'he',
    'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or',
    'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about',
    'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
    'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than',
    'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
    'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give',
    'day', 'most', 'us', 'hello', 'world', 'thanks', 'bye', 'yes', 'no', 'please'
]

# Frequency map to bias ranking (extend if required)
WORD_FREQ = {
    'the': 10000, 'be': 8000, 'to': 8000, 'of': 7600, 'and': 7500, 'a': 7400,
    'in': 7200, 'that': 4800, 'have': 4600, 'i': 4500, 'it': 4400, 'for': 4300,
    'not': 4200, 'on': 4100, 'with': 4000, 'he': 3900, 'as': 3800, 'you': 3700,
    'do': 3600, 'at': 3500, 'this': 3400, 'but': 3300, 'hello': 2500, 'world': 2400,
    'thanks': 2300, 'yes': 2200, 'no': 2100, 'please': 2000, 'good': 3000, 'how': 3000,
}
for _w in FALLBACK_WORDS:
    lw = _w.lower()
    if lw not in WORD_FREQ:
        WORD_FREQ[lw] = 50

# How many suggestion buttons to display
SUGGESTION_COUNT = 8


class SignLanguagePredictor:
    def __init__(self, model_dir=config.MODEL_DIR):
        self.model_dir = model_dir
        self.loaded_model = None
        self.loaded_model_dru = None
        self._load_models()
        self.ct = {char: 0 for char in list(ascii_uppercase) + ['blank']}
        self.history = []
        self.char_accepted_flag = False
        
    def _load_models(self):
        def load_single_model(json_path, weights_path):
            if not os.path.exists(json_path) or not os.path.exists(weights_path):
                raise FileNotFoundError(f"Missing model files: {json_path} or {weights_path}")
            with open(json_path, 'r') as jf:
                model_json = jf.read()
            model = model_from_json(model_json)
            model.load_weights(weights_path)
            return model

        try:
            self.loaded_model = load_single_model(config.MODEL_BW_JSON, config.MODEL_BW_H5)
            self.loaded_model_dru = load_single_model(config.MODEL_BW_DRU_JSON, config.MODEL_BW_DRU_H5)
            print("All models loaded successfully.")
        except Exception as e:
            print(f"FATAL: Model loading failed. Error: {e}")
            # We might want to handle this gracefully in UI, but for now raise
            raise

    def predict(self, test_image):
        if test_image is None or getattr(test_image, "size", 0) == 0:
            return None, 0.0

        test_image = cv2.resize(test_image, (config.IMG_SIZE, config.IMG_SIZE))
        arr = test_image.reshape(1, config.IMG_SIZE, config.IMG_SIZE, 1).astype('float32') / 255.0

        result = self.loaded_model.predict(arr, verbose=0)
        prediction = {ascii_uppercase[i]: float(result[0][i + 1]) for i in range(26)}
        prediction['blank'] = float(result[0][0])

        prediction_sorted = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        top_symbol = prediction_sorted[0][0]
        top_prob = prediction_sorted[0][1]

        if top_symbol in ('D', 'R', 'U'):
            result_dru = self.loaded_model_dru.predict(arr, verbose=0)
            pred2 = {'D': float(result_dru[0][0]), 'R': float(result_dru[0][1]), 'U': float(result_dru[0][2])}
            top_symbol = max(pred2, key=pred2.get)
            top_prob = pred2[top_symbol]

        return top_symbol, top_prob

    def process_prediction(self, top_symbol):
        THRESHOLD_COUNT = 15
        confirmed_char = None
        
        if top_symbol != 'blank':
            self.ct[top_symbol] += 1
            if self.ct[top_symbol] > THRESHOLD_COUNT and not self.char_accepted_flag:
                if len(self.history) == 0 or self.history[-1] != top_symbol:
                    confirmed_char = top_symbol
                    self.history.append(top_symbol)
                self.char_accepted_flag = True
        else:
            for char in self.ct:
                self.ct[char] = 0
            self.char_accepted_flag = False
            
        return confirmed_char


class SuggestionEngine:
    def __init__(self, model_dir=config.MODEL_DIR):
        self.model_dir = model_dir
        self.suggestion_engine = None
        self.wordlist = []
        self._init_backend()

    def _init_backend(self):
        if ENCHANT_AVAILABLE:
            try:
                self.suggestion_engine = enchant.Dict('en_US')
                print("Using 'enchant' for candidate suggestions.")
            except Exception as e:
                print(f"enchant available but couldn't initialize: {e}")
                self.suggestion_engine = None

        candidate_wordlist = None

        # 1) custom wordlist in model dir
        wl_path = os.path.join(self.model_dir, 'wordlist.txt')
        if os.path.exists(wl_path):
            try:
                with open(wl_path, 'r', encoding='utf-8') as f:
                    candidate_wordlist = [w.strip().lower() for w in f if w.strip()]
                print(f"Loaded custom wordlist from {wl_path} ({len(candidate_wordlist)} words).")
            except Exception as e:
                print(f"Failed to read {wl_path}: {e}")
                candidate_wordlist = None

        # 2) wordfreq
        if candidate_wordlist is None and WORDFREQ_AVAILABLE:
            try:
                wf = top_n_list("en", n_top=100000) # Increased to 100k
                if wf:
                    candidate_wordlist = [w.lower() for w in wf]
                    print(f"Built wordlist from wordfreq (size {len(candidate_wordlist)}).")
            except Exception:
                try:
                    wf = top_n_list("en", 50000)
                    candidate_wordlist = [w.lower() for w in wf]
                    print(f"Built wordlist from wordfreq (size {len(candidate_wordlist)}).")
                except Exception:
                    candidate_wordlist = None

        # 3) system dict
        if candidate_wordlist is None:
            system_paths = ['/usr/share/dict/words', '/usr/dict/words', '/usr/dict/web2', '/usr/dict/web2a']
            for p in system_paths:
                if os.path.exists(p):
                    try:
                        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                            candidate_wordlist = [w.strip().lower() for w in f if w.strip() and w.strip().isalpha()]
                        print(f"Loaded system dictionary from {p} ({len(candidate_wordlist)} words).")
                        break
                    except Exception as e:
                        print(f"Failed to load system dict {p}: {e}")

        # 4) fallback - Expanded list
        if candidate_wordlist is None:
            # A slightly larger fallback list to ensure we have something decent
            EXTRA_FALLBACK = [
                'about', 'above', 'across', 'action', 'activity', 'actually', 'add', 'address', 'administration',
                'admit', 'adult', 'affect', 'after', 'again', 'against', 'age', 'agency', 'agent', 'ago', 'agree',
                'agreement', 'ahead', 'air', 'all', 'allow', 'almost', 'alone', 'along', 'already', 'also', 'although',
                'always', 'american', 'among', 'amount', 'analysis', 'and', 'animal', 'another', 'answer', 'any',
                'anyone', 'anything', 'appear', 'apply', 'approach', 'area', 'argue', 'arm', 'around', 'arrive', 'art',
                'article', 'artist', 'as', 'ask', 'assume', 'at', 'attack', 'attention', 'attorney', 'audience',
                'author', 'authority', 'available', 'avoid', 'away', 'baby', 'back', 'bad', 'bag', 'ball', 'bank',
                'bar', 'base', 'be', 'beat', 'beautiful', 'because', 'become', 'bed', 'before', 'begin', 'behavior',
                'behind', 'believe', 'benefit', 'best', 'better', 'between', 'beyond', 'big', 'bill', 'billion', 'bit',
                'black', 'blood', 'blue', 'board', 'body', 'book', 'born', 'both', 'box', 'boy', 'break', 'bring',
                'brother', 'budget', 'build', 'building', 'business', 'but', 'buy', 'by', 'call', 'camera', 'campaign',
                'can', 'cancer', 'candidate', 'capital', 'car', 'card', 'care', 'career', 'carry', 'case', 'catch',
                'cause', 'cell', 'center', 'central', 'century', 'certain', 'certainly', 'chair', 'challenge', 'chance',
                'change', 'character', 'charge', 'check', 'child', 'choice', 'choose', 'church', 'citizen', 'city',
                'civil', 'claim', 'class', 'clear', 'clearly', 'close', 'coach', 'cold', 'collection', 'college',
                'color', 'come', 'commercial', 'common', 'community', 'company', 'compare', 'computer', 'concern',
                'condition', 'conference', 'congress', 'consider', 'consumer', 'contain', 'continue', 'control', 'cost',
                'could', 'country', 'couple', 'course', 'court', 'cover', 'create', 'crime', 'cultural', 'culture',
                'cup', 'current', 'customer', 'cut', 'dark', 'data', 'daughter', 'day', 'dead', 'deal', 'death',
                'debate', 'decade', 'decide', 'decision', 'deep', 'defense', 'degree', 'democrat', 'democratic',
                'describe', 'design', 'despite', 'detail', 'determine', 'develop', 'development', 'die', 'difference',
                'different', 'difficult', 'dinner', 'direction', 'director', 'discover', 'discuss', 'discussion',
                'disease', 'do', 'doctor', 'dog', 'door', 'down', 'draw', 'dream', 'drive', 'drop', 'drug', 'during',
                'each', 'early', 'east', 'easy', 'eat', 'economic', 'economy', 'edge', 'education', 'effect', 'effort',
                'eight', 'either', 'election', 'else', 'employee', 'end', 'energy', 'enjoy', 'enough', 'enter',
                'entire', 'environment', 'environmental', 'especially', 'establish', 'even', 'evening', 'event', 'ever',
                'every', 'everybody', 'everyone', 'everything', 'evidence', 'exactly', 'example', 'executive', 'exist',
                'expect', 'experience', 'expert', 'explain', 'eye', 'face', 'fact', 'factor', 'fail', 'fall', 'family',
                'far', 'fast', 'father', 'fear', 'federal', 'feel', 'feeling', 'few', 'field', 'fight', 'figure',
                'fill', 'film', 'final', 'finally', 'financial', 'find', 'fine', 'finger', 'finish', 'fire', 'firm',
                'first', 'fish', 'five', 'floor', 'fly', 'focus', 'follow', 'food', 'foot', 'for', 'force', 'foreign',
                'forget', 'form', 'former', 'forward', 'four', 'free', 'friend', 'from', 'front', 'full', 'fund',
                'future', 'game', 'garden', 'gas', 'general', 'generation', 'get', 'girl', 'give', 'glass', 'go',
                'goal', 'good', 'government', 'great', 'green', 'ground', 'group', 'grow', 'growth', 'guess', 'gun',
                'guy', 'hair', 'half', 'hand', 'hang', 'happen', 'happy', 'hard', 'have', 'he', 'head', 'health',
                'hear', 'heart', 'heat', 'heavy', 'help', 'her', 'here', 'herself', 'high', 'him', 'himself', 'his',
                'history', 'hit', 'hold', 'home', 'hope', 'hospital', 'hot', 'hotel', 'hour', 'house', 'how', 'however',
                'huge', 'human', 'hundred', 'husband', 'i', 'idea', 'identify', 'if', 'image', 'imagine', 'impact',
                'important', 'improve', 'in', 'include', 'including', 'increase', 'indeed', 'indicate', 'individual',
                'industry', 'information', 'inside', 'instead', 'institution', 'interest', 'interesting', 'international',
                'interview', 'into', 'investment', 'involve', 'issue', 'it', 'item', 'its', 'itself', 'job', 'join',
                'just', 'keep', 'key', 'kid', 'kill', 'kind', 'kitchen', 'know', 'knowledge', 'land', 'language',
                'large', 'last', 'late', 'later', 'laugh', 'law', 'lawyer', 'lay', 'lead', 'leader', 'learn', 'least',
                'leave', 'left', 'leg', 'legal', 'less', 'let', 'letter', 'level', 'lie', 'life', 'light', 'like',
                'likely', 'line', 'list', 'listen', 'little', 'live', 'local', 'long', 'look', 'lose', 'loss', 'lot',
                'love', 'low', 'machine', 'magazine', 'main', 'maintain', 'major', 'majority', 'make', 'man', 'manage',
                'management', 'manager', 'many', 'market', 'marriage', 'material', 'matter', 'may', 'maybe', 'me',
                'mean', 'measure', 'media', 'medical', 'meet', 'meeting', 'member', 'memory', 'mention', 'message',
                'method', 'middle', 'might', 'military', 'million', 'mind', 'minute', 'miss', 'mission', 'model',
                'modern', 'moment', 'money', 'month', 'more', 'morning', 'most', 'mother', 'mouth', 'move', 'movement',
                'movie', 'mr', 'mrs', 'much', 'music', 'must', 'my', 'myself', 'name', 'nation', 'national', 'natural',
                'nature', 'near', 'nearly', 'necessary', 'need', 'network', 'never', 'new', 'news', 'newspaper', 'next',
                'nice', 'night', 'no', 'none', 'nor', 'north', 'not', 'note', 'nothing', 'notice', 'now', 'number',
                'occur', 'of', 'off', 'offer', 'office', 'officer', 'official', 'often', 'oh', 'oil', 'ok', 'old', 'on',
                'once', 'one', 'only', 'onto', 'open', 'operation', 'opportunity', 'option', 'or', 'order',
                'organization', 'other', 'others', 'our', 'out', 'outside', 'over', 'own', 'owner', 'page', 'pain',
                'painting', 'paper', 'parent', 'part', 'participant', 'particular', 'particularly', 'partner', 'party',
                'pass', 'past', 'patient', 'pattern', 'pay', 'peace', 'people', 'per', 'perform', 'performance',
                'perhaps', 'period', 'person', 'personal', 'phone', 'physical', 'pick', 'picture', 'piece', 'place',
                'plan', 'plant', 'play', 'player', 'pm', 'point', 'police', 'policy', 'political', 'politics', 'poor',
                'popular', 'population', 'position', 'positive', 'possible', 'power', 'practice', 'prepare', 'present',
                'president', 'pressure', 'pretty', 'prevent', 'price', 'private', 'probably', 'problem', 'process',
                'produce', 'product', 'production', 'professional', 'professor', 'program', 'project', 'property',
                'protect', 'prove', 'provide', 'public', 'pull', 'purpose', 'push', 'put', 'quality', 'question',
                'quickly', 'quite', 'race', 'radio', 'raise', 'range', 'rate', 'rather', 'reach', 'read', 'ready',
                'real', 'reality', 'realize', 'really', 'reason', 'receive', 'recent', 'recently', 'recognize',
                'record', 'red', 'reduce', 'reflect', 'region', 'relate', 'relationship', 'religious', 'remain',
                'remember', 'remove', 'report', 'represent', 'republican', 'require', 'research', 'resource', 'respond',
                'response', 'responsibility', 'rest', 'result', 'return', 'reveal', 'rich', 'right', 'rise', 'risk',
                'road', 'rock', 'role', 'room', 'rule', 'run', 'safe', 'same', 'save', 'say', 'scene', 'school',
                'science', 'scientist', 'score', 'sea', 'season', 'seat', 'second', 'section', 'security', 'see',
                'seek', 'seem', 'sell', 'send', 'senior', 'sense', 'series', 'serious', 'serve', 'service', 'set',
                'seven', 'several', 'sex', 'sexual', 'shake', 'share', 'she', 'shoot', 'short', 'shot', 'should',
                'shoulder', 'show', 'side', 'sign', 'significant', 'similar', 'simple', 'simply', 'since', 'sing',
                'single', 'sister', 'sit', 'site', 'situation', 'six', 'size', 'skill', 'skin', 'small', 'smile', 'so',
                'social', 'society', 'soldier', 'some', 'somebody', 'someone', 'something', 'sometimes', 'son', 'song',
                'soon', 'sort', 'sound', 'source', 'south', 'southern', 'space', 'speak', 'special', 'specific',
                'speech', 'spend', 'sport', 'spring', 'staff', 'stage', 'stand', 'standard', 'star', 'start', 'state',
                'statement', 'station', 'stay', 'step', 'still', 'stock', 'stop', 'store', 'story', 'strategy',
                'street', 'strong', 'structure', 'student', 'study', 'stuff', 'style', 'subject', 'success',
                'successful', 'such', 'suddenly', 'suffer', 'suggest', 'summer', 'support', 'sure', 'surface', 'system',
                'table', 'take', 'talk', 'task', 'tax', 'teach', 'teacher', 'team', 'technology', 'television', 'tell',
                'ten', 'tend', 'term', 'test', 'than', 'thank', 'that', 'the', 'their', 'them', 'themselves', 'then',
                'theory', 'there', 'these', 'they', 'thing', 'think', 'third', 'this', 'those', 'though', 'thought',
                'thousand', 'threat', 'three', 'through', 'throughout', 'throw', 'thus', 'time', 'to', 'today',
                'together', 'tonight', 'too', 'top', 'total', 'tough', 'toward', 'town', 'trade', 'traditional',
                'training', 'travel', 'treat', 'treatment', 'tree', 'trial', 'trip', 'trouble', 'true', 'truth', 'try',
                'turn', 'tv', 'two', 'type', 'under', 'understand', 'unit', 'until', 'up', 'upon', 'us', 'use',
                'usually', 'value', 'various', 'very', 'victim', 'view', 'violence', 'visit', 'voice', 'vote', 'wait',
                'walk', 'wall', 'want', 'war', 'watch', 'water', 'way', 'we', 'weapon', 'wear', 'week', 'weight',
                'well', 'west', 'western', 'what', 'whatever', 'when', 'where', 'whether', 'which', 'while', 'white',
                'who', 'whole', 'whom', 'whose', 'why', 'wide', 'wife', 'will', 'win', 'wind', 'window', 'wish', 'with',
                'within', 'without', 'woman', 'wonder', 'word', 'work', 'worker', 'world', 'worry', 'would', 'write',
                'writer', 'wrong', 'yard', 'yeah', 'year', 'yes', 'yet', 'you', 'young', 'your', 'yourself'
            ]
            candidate_wordlist = [w.lower() for w in FALLBACK_WORDS + EXTRA_FALLBACK]
            print("Using fallback expanded wordlist.")

        # deduplicate & keep reasonable size
        unique_words = list(dict.fromkeys(candidate_wordlist))
        if len(unique_words) > 100000:
            unique_words = unique_words[:100000]
        self.wordlist = sorted(unique_words)
        print(f"Suggestion backend ready. Wordlist size: {len(self.wordlist)}")

    def get_suggestions(self, prefix):
        if not prefix:
            return []
            
        prefix = prefix.lower()
        enchant_candidates = []
        if self.suggestion_engine is not None:
            try:
                enchant_candidates = [w.lower() for w in self.suggestion_engine.suggest(prefix) if w]
            except Exception:
                enchant_candidates = []

        prefix_matches = [w for w in self.wordlist if w.startswith(prefix)]
        fuzzy_candidates = difflib.get_close_matches(prefix, self.wordlist, n=50, cutoff=0.6)

        candidates = []
        for src in (prefix_matches, enchant_candidates, fuzzy_candidates):
            for w in src:
                if w not in candidates:
                    candidates.append(w)

        if len(candidates) < SUGGESTION_COUNT:
            contains = [w for w in self.wordlist if prefix in w and w not in candidates]
            candidates.extend(contains)

        if len(candidates) < SUGGESTION_COUNT and len(prefix) <= 2:
            extras = [w for w in self.wordlist if w.startswith(prefix[:1]) and w not in candidates]
            candidates.extend(extras)

        return self._rank_candidates(prefix, candidates)

    def _rank_candidates(self, prefix, candidates, max_results=SUGGESTION_COUNT):
        scored = []
        for w in candidates:
            lw = w.lower()
            score = 0.0
            if lw.startswith(prefix):
                score += 100.0
                score += len(prefix) * 1.0
            if prefix in lw:
                score += 10.0
            ratio = difflib.SequenceMatcher(None, prefix, lw).ratio()
            score += ratio * 20.0
            freq = WORD_FREQ.get(lw, 1)
            score += min(50.0, (freq ** 0.5) / 2.0)
            scored.append((score, lw))

        scored.sort(key=operator.itemgetter(0), reverse=True)
        result = []
        for s, w in scored:
            if w not in result:
                result.append(w)
            if len(result) >= max_results:
                break
        return result


class Application:
    def __init__(self):
        self.predictor = SignLanguagePredictor()
        self.suggester = SuggestionEngine()
        
        self.vs = cv2.VideoCapture(0)
        
        self.sentence = ""
        self.word = ""
        self.current_symbol = "..."
        self.confidence = 0.0
        
        # UI
        self._setup_tk_root()
        self._setup_theme_and_fonts()
        self._setup_ui_layout()
        self._bind_keyboard_shortcuts()
        self.root.after(200, self._load_and_display_signs_image)
        
        # start
        self.video_loop()

    def _setup_tk_root(self):
        self.root = tk.Tk()
        self.root.title(config.WINDOW_TITLE)
        self.root.configure(bg=config.THEME_COLOR)
        try:
            self.root.state('zoomed')
        except Exception:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)

    def _setup_theme_and_fonts(self):
        self.BG_COLOR = config.THEME_COLOR
        self.TEXT_COLOR = config.TEXT_COLOR
        self.ACCENT_COLOR = config.ACCENT_COLOR
        self.PANEL_BG_COLOR = config.PANEL_BG_COLOR
        self.BUTTON_COLOR = config.BUTTON_COLOR

        self.large_font = tkfont.Font(family="Segoe UI", size=24, weight="bold")
        self.medium_font = tkfont.Font(family="Segoe UI", size=16)
        self.small_font = tkfont.Font(family="Segoe UI", size=12)
        self.hud_font = tkfont.Font(family="Consolas", size=11)

    def _setup_ui_layout(self):
        main_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        left_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Floral Header (if available)
        self.floral_panel = tk.Label(left_frame, bg=self.BG_COLOR)
        self.floral_panel.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        self._load_floral_image()

        self.panel = tk.Label(left_frame, bg=self.PANEL_BG_COLOR)
        self.panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        text_frame = tk.Frame(left_frame, bg=self.BG_COLOR, pady=15)
        text_frame.pack(side=tk.BOTTOM, fill=tk.X)
        text_frame.grid_columnconfigure(1, weight=1)

        tk.Label(text_frame, text="Detected 🌸:", font=self.medium_font, bg=self.BG_COLOR, fg=self.ACCENT_COLOR).grid(row=0, column=0, sticky='nw', pady=2)
        self.current_symbol_label = tk.Label(text_frame, text=self.current_symbol, font=self.large_font, bg=self.BG_COLOR, fg=self.TEXT_COLOR)
        self.current_symbol_label.grid(row=0, column=1, sticky='nw', padx=10)

        # Changed "Word:" to "Letter:" as requested
        tk.Label(text_frame, text="Letter 🌺:", font=self.medium_font, bg=self.BG_COLOR, fg=self.TEXT_COLOR).grid(row=1, column=0, sticky='nw', pady=2)
        self.word_label = tk.Label(text_frame, text=self.word, font=self.medium_font, bg=self.BG_COLOR, fg=self.TEXT_COLOR)
        self.word_label.grid(row=1, column=1, sticky='nw', padx=10)

        tk.Label(text_frame, text="Sentence 🎀:", font=self.medium_font, bg=self.BG_COLOR, fg=self.TEXT_COLOR).grid(row=2, column=0, sticky='nw', pady=2)
        self.sentence_label = tk.Label(text_frame, text=self.sentence, font=self.medium_font, bg=self.BG_COLOR, fg=self.TEXT_COLOR, wraplength=int(self.root.winfo_screenwidth() * 0.5), justify='left')
        self.sentence_label.grid(row=2, column=1, sticky='nw', padx=10)

        self.hud_label = tk.Label(text_frame, text="", font=self.hud_font, bg=self.BG_COLOR, fg='#555555', justify='left')
        self.hud_label.grid(row=3, column=0, columnspan=2, sticky='nw', pady=(20, 0))

        right_frame = tk.Frame(main_frame, bg=self.BG_COLOR, width=380)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        right_frame.pack_propagate(False)

        self.panel2 = tk.Label(right_frame, bg=self.PANEL_BG_COLOR)
        self.panel2.pack(side=tk.TOP, fill=tk.X)

        sugg_frame = tk.Frame(right_frame, bg=self.BG_COLOR)
        sugg_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        tk.Label(sugg_frame, text="Suggestions ✨", font=self.medium_font, bg=self.BG_COLOR, fg=self.ACCENT_COLOR).pack()

        # Create SUGGESTION_COUNT suggestion buttons
        self.suggestion_buttons = []
        for i in range(SUGGESTION_COUNT):
            btn = tk.Button(sugg_frame, text="", font=self.small_font, bg=self.BUTTON_COLOR, fg=self.TEXT_COLOR,
                            relief='flat', command=lambda idx=i: self._use_suggestion(idx))
            btn.pack(fill='x', padx=5, pady=3)
            self.suggestion_buttons.append(btn)

        self.signs_panel = tk.Label(right_frame, bg=self.BG_COLOR)
        self.signs_panel.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def _load_floral_image(self):
        # Try to load floral decoration
        paths = [
            os.path.join(config.BASE_DIR, 'floral_decoration.png'),
            'floral_decoration.png'
        ]
        for p in paths:
            if os.path.exists(p):
                try:
                    img = Image.open(p).convert('RGBA')
                    # Resize to fit width, keep aspect ratio
                    target_w = 400 # reasonable width
                    ratio = target_w / img.width
                    target_h = int(img.height * ratio)
                    if target_h > 100: # Cap height
                        target_h = 100
                        target_w = int(img.width * (100 / img.height))
                    
                    img.thumbnail((target_w, target_h), Image.LANCZOS)
                    
                    # Create a background for it matching the theme
                    bg_img = Image.new('RGBA', (target_w, target_h), self.BG_COLOR)
                    bg_img.paste(img, (0, 0), img)
                    
                    self.floral_img_tk = ImageTk.PhotoImage(bg_img)
                    self.floral_panel.config(image=self.floral_img_tk)
                    print(f"Loaded floral decoration from {p}")
                    return
                except Exception as e:
                    print(f"Failed to load floral image: {e}")

    def _bind_keyboard_shortcuts(self):
        self.root.bind('<space>', self._commit_word)
        self.root.bind('<BackSpace>', self._delete_char)
        self.root.bind('c', self._clear_sentence)
        self.root.bind('<Escape>', lambda e: self.destructor())

    def _load_and_display_signs_image(self):
        candidate_path = 'signs.png'
        if not os.path.exists(candidate_path):
            candidate_path = os.path.join(config.BASE_DIR, 'signs.png')

        if os.path.exists(candidate_path):
            try:
                self.signs_panel.update_idletasks()
                pw = self.signs_panel.winfo_width()
                ph = self.signs_panel.winfo_height()

                if pw > 1 and ph > 1:
                    img = Image.open(candidate_path).convert('RGBA')
                    img.thumbnail((pw, ph), Image.LANCZOS)
                    canvas = Image.new('RGBA', (pw, ph), (0, 0, 0, 255))
                    x = (pw - img.width) // 2
                    y = (ph - img.height) // 2
                    canvas.paste(img, (x, y), img)
                    self.signs_image_tk = ImageTk.PhotoImage(canvas)
                    self.signs_panel.config(image=self.signs_image_tk)
            except Exception as e:
                print(f"Error loading signs.png: {e}")
        else:
            print("Warning: 'signs.png' not found.")

    def _commit_word(self, event=None):
        if self.word:
            self.sentence += (" " if self.sentence else "") + self.word
            self.word = ""
            self._update_text_labels()

    def _delete_char(self, event=None):
        if self.word:
            self.word = self.word[:-1]
            self._update_text_labels()

    def _clear_sentence(self, event=None):
        self.sentence = ""
        self.word = ""
        self._update_text_labels()

    def _use_suggestion(self, idx):
        text = self.suggestion_buttons[idx].cget('text')
        if not text:
            return
        chosen = text.strip().upper()
        self.word = chosen
        self._commit_word()

    def video_loop(self):
        ok, frame = self.vs.read()
        if not ok:
            print("Camera feed lost.")
            self.destructor()
            return

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        roi_size = int(min(height, width) * 0.4)
        x1 = width - roi_size - 20
        y1 = 20
        x2 = width - 20
        y2 = y1 + roi_size

        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 230, 118), 2)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, processed_roi = cv2.threshold(th3, config.MIN_VALUE, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Predict
        top_symbol, top_prob = self.predictor.predict(processed_roi)
        self.current_symbol = top_symbol
        self.confidence = top_prob
        
        # Process prediction (accumulate history, etc.)
        confirmed_char = self.predictor.process_prediction(top_symbol)
        if confirmed_char:
            self.word += confirmed_char

        img2 = Image.fromarray(processed_roi)
        imgtk2 = ImageTk.PhotoImage(image=img2)
        self.panel2.imgtk = imgtk2
        self.panel2.config(image=imgtk2)

        self._update_text_labels()
        self._update_suggestions()
        self._update_hud()

        self.root.after(20, self.video_loop)

    def _update_text_labels(self):
        self.current_symbol_label.config(text=self.current_symbol)
        self.word_label.config(text=self.word)
        self.sentence_label.config(text=self.sentence)

    def _update_suggestions(self):
        suggestions = []
        if self.word:
            suggestions = self.suggester.get_suggestions(self.word)

        # Update UI buttons
        for i, btn in enumerate(self.suggestion_buttons):
            if i < len(suggestions):
                display_text = suggestions[i].capitalize()
                btn.config(text=display_text)
            else:
                btn.config(text="")

    def _update_hud(self):
        hist_str = "".join(self.predictor.history[-10:])
        hud_text = (
            f"Confidence: {self.confidence:.2f}   |   History: {hist_str}\n"
            f"[Space] Add Word | [Backspace] Del Char | [c] Clear All"
        )
        self.hud_label.config(text=hud_text)

    def destructor(self):
        print("Closing application...")
        try:
            self.root.destroy()
        except Exception:
            pass
        try:
            self.vs.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        app = Application()
        app.root.mainloop()
    except Exception as e:
        print(f"An error occurred during application startup: {e}")
