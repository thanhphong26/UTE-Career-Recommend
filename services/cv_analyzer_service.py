import os
import re
import logging
import tempfile
import requests
from typing import List, Dict, Any, Optional, Tuple, Union
from sqlalchemy.orm import Session
from datetime import datetime

# Basic libraries
import PyPDF2
import docx
import pandas as pd
import numpy as np

# NLP and ML libraries for enhanced recommendation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

# Define path constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'ml_models')
os.makedirs(MODELS_DIR, exist_ok=True)

class SimpleCVAnalyzer:
    """
    Simplified CV analyzer that works without spaCy and other ML libraries
    """
    
    # Constants and predefined data
    SECTION_KEYWORDS = {
        'education': ['education', 'academic', 'học vấn', 'bằng cấp', 'trường', 'university', 'school', 'college', 'degree'],
        'experience': ['experience', 'work experience', 'employment', 'history', 'kinh nghiệm', 'làm việc', 'work history', 'professional experience'],
        'skills': ['skills', 'technical skills', 'kỹ năng', 'chuyên môn', 'công nghệ', 'technology', 'competencies', 'expertise'],
        'projects': ['projects', 'personal projects', 'academic projects', 'dự án', 'đồ án', 'project experience'],
        'certifications': ['certifications', 'certificates', 'chứng chỉ', 'licenses', 'credentials'],
        'languages': ['languages', 'ngôn ngữ', 'language proficiency', 'foreign languages'],
        'summary': ['summary', 'profile', 'tóm tắt', 'professional summary', 'objective', 'about me', 'personal statement'],
        'contact': ['contact', 'liên hệ', 'thông tin', 'email', 'phone', 'điện thoại', 'địa chỉ', 'address']
    }
    
    TECHNICAL_SKILLS = [
        # Programming languages
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin', 'go', 'rust',
        # Web development
        'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'laravel', 'asp.net',
        'jquery', 'bootstrap', 'tailwind', 'next.js', 'nuxt.js', 'svelte',
        # Mobile
        'android', 'ios', 'flutter', 'react native', 'swift', 'xamarin', 'ionic',
        # Databases
        'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'sqlite', 'redis', 'firebase', 'dynamodb', 'cassandra',
        'mariadb', 'elasticsearch', 'neo4j', 
        # DevOps
        'docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions', 'travis ci', 'ansible', 'terraform',
        'aws', 'azure', 'gcp', 'cloud', 'devops', 'ci/cd', 'continuous integration', 'continuous deployment',
        # Data Science & AI
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'machine learning', 'deep learning', 'nlp', 'natural language processing', 'computer vision',
        'data mining', 'data visualization', 'statistics', 'neural networks', 'reinforcement learning',
        # Other frameworks and tools
        'git', 'spring', 'hibernate', 'junit', 'selenium', 'agile', 'scrum', 'rest api', 'graphql', 
        'microservices', 'jira', 'trello', 'pytest', 'linux', 'bash', 'powershell', 'serverless', 'figma',
        'adobe xd', 'photoshop', 'illustrator'
    ]
    
    INDUSTRY_CATEGORIES = [
        'information technology', 'software development', 'web development', 'mobile development', 
        'data science', 'artificial intelligence', 'machine learning', 'cybersecurity', 'networking',
        'devops', 'cloud computing', 'database administration', 'game development', 'ui/ux design',
        'product management', 'project management', 'quality assurance', 'technical support',
        'digital marketing', 'e-commerce', 'telecommunications', 'healthcare it', 'fintech',
        'edtech', 'business intelligence'
    ]
    
    def __init__(self, temp_upload_dir: str = 'uploads'):
        """
        Initialize the simplified CV analyzer
        
        :param temp_upload_dir: Temporary directory for uploaded files
        """
        self.temp_upload_dir = temp_upload_dir
        os.makedirs(temp_upload_dir, exist_ok=True)
        
        # Convert skill list to lowercase and create a set for efficient lookups
        self.skill_set = set(skill.lower() for skill in self.TECHNICAL_SKILLS)
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF files using PyPDF2
        
        :param file_path: Path to the PDF file
        :return: Extracted text
        """
        text = ""
        
        # Try PyPDF2
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
                
        return text
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text from DOCX files
        
        :param file_path: Path to the DOCX file
        :return: Extracted text
        """
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> Tuple[str, str]:
        """
        Extract text from various file formats
        
        :param file_path: Path to the file
        :return: Tuple of (extracted text, detected language)
        """
        extension = os.path.splitext(file_path)[1].lower()
        text = ""
        
        if extension == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif extension in ['.docx', '.doc']:
            text = self.extract_text_from_docx(file_path)
        else:
            logger.error(f"Unsupported file format: {extension}")
            return "", "unknown"
            
        # Simple language detection (basic approach without langdetect)
        # Just check if the text contains more Vietnamese characters than English
        vietnamese_chars = len(re.findall(r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', text.lower()))
        if vietnamese_chars > len(text) * 0.05:  # If more than 5% are Vietnamese chars
            lang = "vi"
        else:
            lang = "en"  # Default to English
            
        return text, lang
    
    def preprocess_text(self, text: str, language: str = "en") -> str:
        """
        Simple text preprocessing
        
        :param text: Raw text
        :param language: Detected language code
        :return: Preprocessed text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters, keeping only alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_skills(self, text: str, custom_skills: List[str] = None) -> List[str]:
        """
        Extract skills from text using pattern matching
        
        :param text: CV text
        :param custom_skills: Additional skills to look for
        :return: List of identified skills
        """
        if custom_skills:
            # Add custom skills to our skill set
            self.skill_set.update(set(skill.lower() for skill in custom_skills))
            
        extracted_skills = set()
        text_lower = text.lower()
        
        # Pattern matching approach
        for skill in self.skill_set:
            # Find skill as a standalone word (not part of another word)
            pattern = r'\b{}\b'.format(re.escape(skill))
            if re.search(pattern, text_lower):
                extracted_skills.add(skill)
        
        return sorted(list(extracted_skills))
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract different sections from the CV
        
        :param text: CV text
        :return: Dictionary with sections and their content
        """
        sections = {}
        lines = text.split('\n')
        
        current_section = None
        section_content = ""
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if this line is a section header
            found_section = None
            for section, keywords in self.SECTION_KEYWORDS.items():
                if any(keyword in line_lower for keyword in keywords):
                    found_section = section
                    break
                    
            if found_section:
                # Save previous section if exists
                if current_section and section_content.strip():
                    sections[current_section] = section_content.strip()
                
                # Start new section
                current_section = found_section
                section_content = ""
            elif current_section:
                # Add line to current section
                section_content += line + " "
        
        # Add the last section
        if current_section and section_content.strip():
            sections[current_section] = section_content.strip()
            
        return sections
    
    def extract_education(self, text: str, sections: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """
        Extract education information using regex patterns
        
        :param text: CV text
        :param sections: Previously extracted CV sections
        :return: List of education entries
        """
        education_info = []
        
        # Get education section text
        education_text = ""
        if sections and 'education' in sections:
            education_text = sections['education']
        else:
            # Try to extract education section
            lines = text.split('\n')
            in_education_section = False
            
            for line in lines:
                line_lower = line.lower()
                
                if any(keyword in line_lower for keyword in self.SECTION_KEYWORDS['education']):
                    in_education_section = True
                    continue
                    
                if in_education_section and any(
                    any(keyword in line_lower for keyword in keywords)
                    for section, keywords in self.SECTION_KEYWORDS.items() 
                    if section != 'education'
                ):
                    in_education_section = False
                    
                if in_education_section and line.strip():
                    education_text += line + " "
        
        if education_text:
            try:
                # Use regex patterns to extract education info
                schools = re.findall(r'([A-Za-z\s]+(?:University|College|Institute|School))(?:\s*of\s*[A-Za-z\s]+)?', education_text)
                degrees = re.findall(r'(Bachelor|Master|PhD|Doctorate|B\.S\.|M\.S\.|B\.A\.|M\.A\.|B\.E\.|M\.E\.)', education_text)
                years = re.findall(r'(?:(19|20)\d{2})\s*-\s*(?:(19|20)\d{2})|(19|20)\d{2}', education_text)
                
                # Simple heuristic to create education entries
                for i in range(max(len(schools), len(degrees))):
                    edu = {}
                    if i < len(schools):
                        edu['school'] = schools[i].strip()
                    if i < len(degrees):
                        edu['degree'] = degrees[i].strip()
                    if i < len(years):
                        if isinstance(years[i], tuple):
                            year_parts = [y for y in years[i] if y]
                            if len(year_parts) >= 2:
                                edu['start_year'] = year_parts[0]
                                edu['end_year'] = year_parts[1]
                            elif len(year_parts) == 1:
                                edu['year'] = year_parts[0]
                        else:
                            edu['year'] = years[i]
                            
                    if edu:
                        education_info.append(edu)
            except Exception as e:
                logger.warning(f"Error extracting education information: {e}")
        
        return education_info
    
    def extract_experience(self, text: str, sections: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """
        Extract work experience information using regex patterns
        
        :param text: CV text
        :param sections: Previously extracted CV sections
        :return: List of work experience entries
        """
        experience_info = []
        
        # Get experience section text
        experience_text = ""
        if sections and 'experience' in sections:
            experience_text = sections['experience']
        else:
            # Extract experience section if not provided
            lines = text.split('\n')
            in_experience_section = False
            
            for line in lines:
                line_lower = line.lower()
                
                if any(keyword in line_lower for keyword in self.SECTION_KEYWORDS['experience']):
                    in_experience_section = True
                    continue
                    
                if in_experience_section and any(
                    any(keyword in line_lower for keyword in keywords)
                    for section, keywords in self.SECTION_KEYWORDS.items() 
                    if section != 'experience'
                ):
                    in_experience_section = False
                    
                if in_experience_section and line.strip():
                    experience_text += line + " "
        
        if experience_text:
            try:
                # Use regex to extract experience information
                companies = re.findall(r'(?:[A-Z][a-z]*\s*)+(?:Inc\.|LLC|Ltd\.?|Corporation|Corp\.?|Company|Co\.?)?', experience_text)
                titles = re.findall(r'(Developer|Engineer|Manager|Analyst|Designer|Director|Coordinator|Specialist|Consultant|Intern)', experience_text)
                
                # Extract date ranges
                date_ranges = []
                try:
                    date_ranges = re.findall(r'(?:((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?)\s*(19|20)\d{2})\s*(?:-|–|to)\s*(?:((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?)\s*(19|20)\d{2})|(?:(19|20)\d{2})\s*(?:-|–|to)\s*(?:(19|20)\d{2})', experience_text)
                except Exception as e:
                    logger.warning(f"Error parsing date ranges: {e}")
                
                for i in range(max(len(companies), len(titles))):
                    exp = {}
                    if i < len(companies) and companies[i].strip():
                        exp['company'] = companies[i].strip()
                    if i < len(titles):
                        exp['title'] = titles[i].strip()
                    if i < len(date_ranges):
                        if isinstance(date_ranges[i], tuple):
                            date_parts = [d for d in date_ranges[i] if d]
                            if len(date_parts) >= 4:  # Format: Month Year - Month Year
                                exp['start_date'] = f"{date_parts[0]} {date_parts[1]}"
                                exp['end_date'] = f"{date_parts[2]} {date_parts[3]}"
                            elif len(date_parts) >= 2:  # Format: Year - Year
                                exp['start_date'] = date_parts[0]
                                exp['end_date'] = date_parts[1]
                            
                    if exp:
                        experience_info.append(exp)
            except Exception as e:
                logger.warning(f"Error extracting experience information: {e}")
        
        return experience_info
    
    def extract_summary(self, text: str, sections: Dict[str, str] = None) -> str:
        """
        Extract professional summary from CV
        
        :param text: CV text
        :param sections: Previously extracted CV sections
        :return: Summary text
        """
        if sections and 'summary' in sections:
            return sections['summary']
            
        # Extract summary section
        lines = text.split('\n')
        in_summary_section = False
        summary_text = ""
        
        for line in lines:
            line_lower = line.lower()
            
            if any(keyword in line_lower for keyword in self.SECTION_KEYWORDS['summary']):
                in_summary_section = True
                continue
                
            if in_summary_section and any(
                any(keyword in line_lower for keyword in keywords)
                for section, keywords in self.SECTION_KEYWORDS.items() 
                if section != 'summary'
            ):
                in_summary_section = False
                
            if in_summary_section and line.strip():
                summary_text += line + " "
                
        # If no summary section found, try to extract the first paragraph
        if not summary_text:
            paragraphs = text.split('\n\n')
            if paragraphs:
                # Find first non-empty paragraph that's likely not contact info
                for para in paragraphs:
                    if len(para.split()) > 10 and not any(
                        keyword in para.lower() 
                        for keyword in self.SECTION_KEYWORDS['contact']
                    ):
                        summary_text = para
                        break
        
        return summary_text.strip()
    
    def download_file_from_url(self, url: str) -> Optional[str]:
        """
        Download file from URL and save to a temporary location
        
        :param url: URL of the file to download
        :return: Path to temporary file or None if failed
        """
        try:
            # Extract extension from URL
            file_ext = os.path.splitext(url.split('/')[-1])[1]
            if not file_ext:
                # Default to PDF if no extension
                file_ext = '.pdf'
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, dir=self.temp_upload_dir) as temp_file:
                # Download the file
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Check for HTTP errors
                
                # Write data to temp file
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                    
                # Return path to temp file
                return temp_file.name
                
        except Exception as e:
            logger.error(f"Error downloading file from URL: {e}")
            return None
    
    def analyze_cv(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze CV and extract structured information
        
        :param file_path: Path to CV file or URL
        :return: Dictionary with extracted information
        """
        # Handle URL (download first)
        if file_path.startswith(('http://', 'https://')):
            logger.info(f"Detected URL path: {file_path}, downloading...")
            temp_path = self.download_file_from_url(file_path)
            if not temp_path:
                return {
                    'success': False,
                    'error': f'Failed to download file from URL: {file_path}'
                }
            file_path = temp_path
            logger.info(f"Downloaded to: {file_path}")
        
        # Extract text and detect language
        text, language = self.extract_text(file_path)
        
        # Clean up temp file if downloaded
        if file_path.startswith(self.temp_upload_dir) and os.path.exists(file_path):
            try:
                logger.info(f"Removing temporary file: {file_path}")
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")
        
        if not text:
            return {
                'success': False,
                'error': 'Failed to extract text from CV'
            }
        
        # Preprocess text
        preprocessed_text = self.preprocess_text(text, language)
        
        # Extract CV sections
        sections = self.extract_sections(text)
        
        # Extract various information
        skills = self.extract_skills(text)
        education = self.extract_education(text, sections)
        experience = self.extract_experience(text, sections)
        summary = self.extract_summary(text, sections)
        
        # Create CV data structure
        cv_data = {
            'success': True,
            'skills': skills,
            'education': education,
            'experience': experience,
            'summary': summary,
            'language': language,
            'raw_text': text
        }
        
        # Simple industry categorization based on skills
        industry = "information technology"  # default
        skill_text = " ".join(skills).lower()
        
        # Check for specific industry patterns
        if any(kw in skill_text for kw in ['web', 'frontend', 'backend', 'react', 'angular', 'vue', 'html', 'css']):
            industry = "web development"
        elif any(kw in skill_text for kw in ['android', 'ios', 'flutter', 'react native', 'mobile']):
            industry = "mobile development"
        elif any(kw in skill_text for kw in ['python', 'pandas', 'numpy', 'data', 'statistics', 'analytics']):
            industry = "data science"
        elif any(kw in skill_text for kw in ['machine learning', 'tensorflow', 'pytorch', 'ai', 'deep learning']):
            industry = "artificial intelligence"
        
        cv_data['industry'] = industry
        
        # Simple seniority estimation based on experience
        total_exp_years = 0
        for exp in experience:
            if 'start_date' in exp and 'end_date' in exp:
                try:
                    # Extract years from dates
                    start_year = int(re.search(r'(19|20)\d{2}', exp['start_date']).group(0))
                    
                    # Handle "present" or current year
                    if 'present' in exp['end_date'].lower():
                        end_year = datetime.now().year
                    else:
                        end_year = int(re.search(r'(19|20)\d{2}', exp['end_date']).group(0))
                        
                    years = end_year - start_year
                    if years > 0:
                        total_exp_years += years
                except Exception as e:
                    logger.warning(f"Error calculating experience years: {e}")
        
        # Assign seniority based on years of experience
        if total_exp_years < 1:
            cv_data['seniority'] = "entry-level"
        elif total_exp_years < 3:
            cv_data['seniority'] = "junior"
        elif total_exp_years < 5:
            cv_data['seniority'] = "mid-level"
        elif total_exp_years < 8:
            cv_data['seniority'] = "senior"
        else:
            cv_data['seniority'] = "lead/principal"
        
        return cv_data


class EnhancedCVAnalyzer(SimpleCVAnalyzer):
    """
    Enhanced CV analyzer with advanced NLP features
    """
    
    def __init__(self, temp_upload_dir: str = 'uploads'):
        """
        Initialize the enhanced CV analyzer
        
        :param temp_upload_dir: Temporary directory for uploaded files
        """
        super().__init__(temp_upload_dir)
        
        # Add Vietnamese stopwords if available
        self.stopwords = []
        try:
            stopwords_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', 'vietnamese_stopwords.txt')
            if os.path.exists(stopwords_path):
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    self.stopwords = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(self.stopwords)} Vietnamese stopwords")
        except Exception as e:
            logger.warning(f"Could not load Vietnamese stopwords: {e}")
    
    def preprocess_text(self, text: str, language: str = "en") -> str:
        """
        Enhanced text preprocessing
        
        :param text: Raw text
        :param language: Detected language code
        :return: Preprocessed text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords if language is Vietnamese
        if language == "vi" and self.stopwords:
            words = text.split()
            words = [word for word in words if word not in self.stopwords]
            text = ' '.join(words)
            
        return text
        

class EnhancedCVRecommender:
    """
    Enhanced job recommender using TF-IDF and Cosine Similarity for better job matching
    """
    
    def __init__(self, cv_analyzer: EnhancedCVAnalyzer, match_threshold: float = 0.4):
        """
        Initialize the enhanced CV recommender with TF-IDF capabilities
        
        :param cv_analyzer: EnhancedCVAnalyzer instance
        :param match_threshold: Threshold score for recommending jobs
        """
        self.cv_analyzer = cv_analyzer
        self.match_threshold = match_threshold
        
        # TF-IDF Vectorizer with stop words
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',  # Default to English stop words
            ngram_range=(1, 2),    # Use both unigrams and bigrams
            min_df=2               # Ignore terms that appear in less than 2 documents
        )
        
        # For KNN model (K-Nearest Neighbors)
        self.knn_model = None
        self.job_ids = []
        self.job_data = {}
    
    def get_job_skills(self, job_id: int, db: Session) -> List[str]:
        """
        Get required skills for a specific job
        
        :param job_id: Job ID
        :param db: Database session
        :return: List of skills
        """
        from models.models import JobSkill, Skills
        from sqlalchemy import select, join
        
        # Query job_skills and skills tables
        stmt = select(Skills.skill_name).select_from(
            join(JobSkill, Skills, JobSkill.skill_id == Skills.skill_id)
        ).where(JobSkill.job_id == job_id)
        
        result = db.execute(stmt).all()
        skills = [row[0] for row in result]
        
        return skills
    
    def extract_skills_from_job(self, job_description: str) -> List[str]:
        """
        Extract skills from job description when not available in database
        
        :param job_description: Job description text
        :return: List of skills
        """
        return self.cv_analyzer.extract_skills(job_description)
    
    def build_common_vocabulary(self, job_corpus: List[str], cv_text: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Build a common vocabulary for both job descriptions and CV text,
        then create TF-IDF vectors using this common vocabulary.
        
        :param job_corpus: List of job description texts
        :param cv_text: CV text
        :return: Tuple of (common vocabulary, job vectors, cv vector)
        """
        # Add CV text to corpus for fitting
        all_texts = job_corpus + [cv_text]
        
        # Fit TF-IDF vectorizer on all texts to create a common vocabulary
        self.vectorizer.fit(all_texts)
        
        # Transform corpus to TF-IDF vectors
        tfidf_matrix = self.vectorizer.transform(all_texts)
        
        # Get job vectors and CV vector separately
        job_vectors = tfidf_matrix[:-1]  # All but the last one
        cv_vector = tfidf_matrix[-1]     # The last one is the CV
        
        # Get vocabulary (feature names)
        vocabulary = self.vectorizer.get_feature_names_out()
        
        return vocabulary, job_vectors, cv_vector
    
    def match_cv_to_job(self, 
                        cv_data: Dict[str, Any], 
                        job_desc: str, 
                        job_skills: List[str]) -> Dict[str, Any]:
        """
        Match CV to job description using TF-IDF and Cosine Similarity
        
        :param cv_data: Processed CV data
        :param job_desc: Job description text
        :param job_skills: List of skills required for the job
        :return: Match results with scores and matched skills
        """
        # 1. Skill matching (Jaccard similarity)
        cv_skills = cv_data.get('skills', [])
        cv_skills_set = set(skill.lower() for skill in cv_skills)
        job_skills_set = set(skill.lower() for skill in job_skills)
        
        matched_skills = cv_skills_set.intersection(job_skills_set)
        missing_skills = job_skills_set - cv_skills_set
        
        # Calculate skill match score (Jaccard similarity)
        skill_match_score = len(matched_skills) / len(job_skills_set) if job_skills_set else 0.0
        
        # 2. Content-based similarity using TF-IDF and Cosine Similarity
        # Get CV text
        cv_text = cv_data.get('raw_text', '')
        if not cv_text:
            cv_summary = cv_data.get('summary', '')
            cv_skills_text = ' '.join(cv_skills)
            cv_experience = ' '.join([f"{exp.get('title', '')} {exp.get('company', '')}" 
                                    for exp in cv_data.get('experience', [])])
            cv_text = f"{cv_summary} {cv_skills_text} {cv_experience}"
        
        # Process texts using TF-IDF
        try:
            # Vectorize both texts
            corpus = [job_desc, cv_text]
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity between job description and CV
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception as e:
            logger.error(f"Error calculating TF-IDF cosine similarity: {e}")
            similarity = 0.0
        
        # 3. Combine scores with weights
        skill_weight = 0.6  # Skills are important
        content_weight = 0.4  # Content similarity also matters
        
        combined_score = (skill_match_score * skill_weight) + (similarity * content_weight)
        
        # Prepare result
        result = {
            'match_score': combined_score,
            'skill_match_score': skill_match_score,
            'content_similarity': similarity,
            'matched_skills': list(matched_skills),
            'missing_skills': list(missing_skills),
            'seniority_match': cv_data.get('seniority', 'unknown')
        }
        
        return result
    
    def recommend_jobs_for_cv(
        self, 
        cv_id: int, 
        db: Session, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recommend jobs for a specific CV using TF-IDF and Cosine Similarity
        
        :param cv_id: CV ID
        :param db: Database session
        :param limit: Maximum number of recommendations
        :return: List of recommended jobs with match details
        """
        from models.models import ResumeCV, Job
        
        # Query CV information
        resume = db.query(ResumeCV).filter(ResumeCV.resume_id == cv_id).first()
        if not resume:
            logger.error(f"Resume with ID {cv_id} not found")
            return []
            
        # Analyze CV
        cv_analysis = self.cv_analyzer.analyze_cv(resume.resume_file)
        if not cv_analysis['success']:
            logger.error(f"Failed to analyze CV with ID {cv_id}")
            return []
            
        # Get active jobs
        jobs = db.query(Job).filter(Job.status == "ACTIVE").all()
        logger.info(f"Found {len(jobs)} active jobs to match against")
        
        # Prepare job descriptions and skills
        job_descriptions = []
        job_skills_dict = {}
        
        for job in jobs:
            # Prepare job description
            job_desc = f"{job.job_title} {job.job_description or ''} {job.job_requirements or ''}"
            job_descriptions.append(job_desc)
            
            # Get or extract job skills
            job_skills = self.get_job_skills(job.job_id, db)
            if not job_skills and job.job_requirements:
                job_skills = self.extract_skills_from_job(job.job_requirements)
                
            job_skills_dict[job.job_id] = job_skills
        
        # Get CV text for vector representation
        cv_text = cv_analysis.get('raw_text', '')
        if not cv_text:
            cv_summary = cv_analysis.get('summary', '')
            cv_skills = cv_analysis.get('skills', [])
            cv_skills_text = ' '.join(cv_skills)
            cv_experience = ' '.join([f"{exp.get('title', '')} {exp.get('company', '')}" 
                                    for exp in cv_analysis.get('experience', [])])
            cv_text = f"{cv_summary} {cv_skills_text} {cv_experience}"
        
        # Apply Cosine Similarity with TF-IDF approach as shown in the diagram
        try:
            # Build common vocabulary
            vocabulary, job_vectors, cv_vector = self.build_common_vocabulary(job_descriptions, cv_text)
            
            # Calculate cosine similarity between CV and each job
            similarities = cosine_similarity(cv_vector, job_vectors)[0]
            
            # Create job matches with similarity scores
            job_matches = []
            for i, job in enumerate(jobs):
                job_id = job.job_id
                
                # Get job skills
                job_skills = job_skills_dict.get(job_id, [])
                
                # Calculate skill match score
                cv_skills = cv_analysis.get('skills', [])
                cv_skills_set = set(skill.lower() for skill in cv_skills)
                job_skills_set = set(skill.lower() for skill in job_skills)
                
                matched_skills = cv_skills_set.intersection(job_skills_set)
                missing_skills = job_skills_set - cv_skills_set
                
                skill_match_score = len(matched_skills) / len(job_skills_set) if job_skills_set else 0.0
                
                # Get content similarity from cosine similarity
                content_similarity = similarities[i]
                
                # Calculate combined score
                skill_weight = 0.6
                content_weight = 0.4
                combined_score = (skill_match_score * skill_weight) + (content_similarity * content_weight)
                
                # Add to matches if above threshold
                if combined_score >= self.match_threshold:
                    match_result = {
                        'match_score': combined_score,
                        'skill_match_score': skill_match_score,
                        'content_similarity': content_similarity,
                        'matched_skills': list(matched_skills),
                        'missing_skills': list(missing_skills),
                    }
                    
                    # Create job match entry
                    job_data = {
                        'job_id': job.job_id,
                        'job_title': job.job_title,
                        'match_score': combined_score,
                        'skill_match_score': skill_match_score,
                        'content_similarity': content_similarity,
                        'matched_skills': list(matched_skills),
                        'missing_skills': list(missing_skills),
                        'employer_id': job.employer_id,
                        'category_id': job.category_id,
                        'level_id': job.level_id,
                        'reason': self._generate_match_reason(match_result)
                    }
                    job_matches.append(job_data)
                    
        except Exception as e:
            logger.error(f"Error in TF-IDF recommendation process: {e}")
            # Fallback to simple matching if TF-IDF fails
            job_matches = []
            for job in jobs:
                # Get or extract job skills
                job_skills = job_skills_dict.get(job.job_id, [])
                
                # Prepare job description
                job_desc = f"{job.job_title} {job.job_description or ''} {job.job_requirements or ''}"
                
                # Match CV to job with simple matching (fallback)
                match_result = self._simple_match_cv_to_job(
                    cv_analysis, 
                    job_desc, 
                    job_skills
                )
                
                # Skip jobs below threshold
                if match_result['match_score'] >= self.match_threshold:
                    # Create job match entry
                    job_data = {
                        'job_id': job.job_id,
                        'job_title': job.job_title,
                        'match_score': match_result['match_score'],
                        'skill_match_score': match_result['skill_match_score'],
                        'content_similarity': match_result['content_similarity'],
                        'matched_skills': match_result['matched_skills'],
                        'missing_skills': match_result['missing_skills'],
                        'employer_id': job.employer_id,
                        'category_id': job.category_id,
                        'level_id': job.level_id,
                        'reason': self._generate_match_reason(match_result)
                    }
                    job_matches.append(job_data)
        
        # Sort by match score and limit results
        sorted_matches = sorted(job_matches, key=lambda x: x['match_score'], reverse=True)[:limit]
        
        return sorted_matches
    
    def _simple_match_cv_to_job(self, 
                               cv_data: Dict[str, Any], 
                               job_desc: str, 
                               job_skills: List[str]) -> Dict[str, Any]:
        """
        Simple fallback method for CV-job matching
        
        :param cv_data: Processed CV data
        :param job_desc: Job description text
        :param job_skills: List of skills required for the job
        :return: Match results with scores and matched skills
        """
        # 1. Skill matching (Jaccard similarity)
        cv_skills = cv_data.get('skills', [])
        cv_skills_set = set(skill.lower() for skill in cv_skills)
        job_skills_set = set(skill.lower() for skill in job_skills)
        
        matched_skills = cv_skills_set.intersection(job_skills_set)
        missing_skills = job_skills_set - cv_skills_set
        
        # Calculate skill match score (Jaccard similarity)
        skill_match_score = len(matched_skills) / len(job_skills_set) if job_skills_set else 0.0
        
        # 2. Simple content similarity (term overlap)
        cv_text = cv_data.get('raw_text', '').lower()
        if not cv_text:
            cv_summary = cv_data.get('summary', '').lower()
            cv_skills_text = ' '.join(cv_skills).lower()
            cv_experience = ' '.join([f"{exp.get('title', '')} {exp.get('company', '')}" 
                                    for exp in cv_data.get('experience', [])]).lower()
            cv_text = f"{cv_summary} {cv_skills_text} {cv_experience}"
        
        job_desc_lower = job_desc.lower()
        job_words = set(job_desc_lower.split())
        cv_words = set(cv_text.split())
        
        word_overlap = len(job_words.intersection(cv_words)) / len(job_words) if job_words else 0.0
        
        # 3. Combine scores (weighted average)
        skill_weight = 0.7
        content_weight = 0.3
        
        combined_score = (skill_match_score * skill_weight) + (word_overlap * content_weight)
        
        # Prepare result
        result = {
            'match_score': combined_score,
            'skill_match_score': skill_match_score,
            'content_similarity': word_overlap,
            'matched_skills': list(matched_skills),
            'missing_skills': list(missing_skills),
            'seniority_match': cv_data.get('seniority', 'unknown')
        }
        
        return result
    
    def recommend_jobs_for_student(
        self, 
        student_id: int, 
        db: Session, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recommend jobs for a student based on their latest CV
        
        :param student_id: Student ID
        :param db: Database session
        :param limit: Maximum number of recommendations
        :return: List of recommended jobs with match details
        """
        from models.models import ResumeCV
        from sqlalchemy import desc
        
        # Find latest active CV for the student
        latest_resume = db.query(ResumeCV).filter(
            ResumeCV.student_id == student_id,
            ResumeCV.is_active == 1
        ).order_by(desc(ResumeCV.updated_at)).first()
        
        if not latest_resume:
            logger.error(f"No active resume found for student with ID {student_id}")
            return []
            
        # Use CV recommendation
        return self.recommend_jobs_for_cv(
            latest_resume.resume_id, 
            db, 
            limit
        )
    
    def _generate_match_reason(self, match_result: Dict[str, Any]) -> str:
        """
        Generate human-readable match reason based on results
        
        :param match_result: Match result dictionary
        :return: Explanation string
        """
        match_score = match_result['match_score']
        skill_match = match_result['skill_match_score']
        content_similarity = match_result['content_similarity']
        matched_skills = match_result['matched_skills']
        
        # Generate reason based on the dominant matching factor
        if skill_match > content_similarity:
            # Skills are the dominant factor
            if match_score > 0.8:
                base = "Hồ sơ của bạn rất phù hợp với vị trí này dựa trên các kỹ năng chuyên môn"
            elif match_score > 0.6:
                base = "Kỹ năng chuyên môn của bạn phù hợp tốt với yêu cầu vị trí này"
            elif match_score > 0.4:
                base = "Bạn có một số kỹ năng quan trọng mà vị trí này cần"
            else:
                base = "Bạn có một vài kỹ năng phù hợp với vị trí này"
        else:
            # Content similarity is the dominant factor
            if match_score > 0.8:
                base = "Hồ sơ và kinh nghiệm của bạn rất phù hợp với mô tả công việc này"
            elif match_score > 0.6:
                base = "Kinh nghiệm và nội dung hồ sơ của bạn phù hợp với công việc này"
            elif match_score > 0.4:
                base = "Có sự tương đồng giữa hồ sơ của bạn và yêu cầu công việc này"
            else:
                base = "Có một số điểm tương đồng giữa hồ sơ của bạn và công việc này"
        
        # Add skill match details
        if matched_skills and len(matched_skills) > 0:
            # Limit to 3 skills for brevity
            skill_list = ", ".join(matched_skills[:3])
            if len(matched_skills) > 3:
                skill_list += f" và {len(matched_skills) - 3} kỹ năng khác"
                
            base += f". Bạn có các kỹ năng phù hợp bao gồm: {skill_list}"
        
        # Add recommendation for missing skills if applicable
        missing_skills = match_result['missing_skills']
        if missing_skills and len(missing_skills) > 0 and match_score > 0.4:
            # Limit to 2 missing skills
            missing_list = ", ".join(missing_skills[:2])
            if len(missing_skills) > 2:  # Fixed: comparing length of list with integer
                missing_list += f" và {len(missing_skills) - 2} kỹ năng khác"
                
            base += f". Cải thiện {missing_list} sẽ giúp bạn phù hợp hơn"
        
        return base
        
    def build_knn_model(self, db: Session) -> None:
        """
        Build a K-Nearest Neighbors model on all job data
        
        :param db: Database session
        """
        from models.models import Job
        
        # Get all active jobs
        jobs = db.query(Job).filter(Job.status == "ACTIVE").all()
        
        # Skip if no jobs available
        if not jobs:
            logger.warning("No active jobs found for KNN model")
            return
            
        # Prepare job descriptions
        job_descriptions = []
        job_ids = []
        job_data = {}
        
        for job in jobs:
            # Prepare job description
            job_desc = f"{job.job_title} {job.job_description or ''} {job.job_requirements or ''}"
            
            job_descriptions.append(job_desc)
            job_ids.append(job.job_id)
            
            # Store job data for later use
            job_data[job.job_id] = {
                'job_id': job.job_id,
                'job_title': job.job_title,
                'employer_id': job.employer_id,
                'category_id': job.category_id,
                'level_id': job.level_id
            }
        
        try:
            # Vectorize job descriptions using TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(job_descriptions)
            
            # Create KNN model
            self.knn_model = NearestNeighbors(
                n_neighbors=min(10, len(job_ids)),  # Limit neighbors to available jobs
                algorithm='auto',
                metric='cosine'
            ).fit(tfidf_matrix)
            
            # Save job IDs and data for later retrieval
            self.job_ids = job_ids
            self.job_data = job_data
            
            logger.info(f"Built KNN model with {len(job_ids)} jobs")
        except Exception as e:
            logger.error(f"Error building KNN model: {e}")
    
    def recommend_jobs_with_knn(
        self,
        cv_text: str,
        db: Session,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recommend jobs using KNN approach
        
        :param cv_text: CV text
        :param db: Database session
        :param limit: Maximum number of recommendations
        :return: List of recommended jobs
        """
        # Build or update KNN model if needed
        if self.knn_model is None or not self.job_ids:
            self.build_knn_model(db)
        
        # If still no model, return empty list
        if self.knn_model is None:
            logger.error("Could not build KNN model")
            return []
        
        try:
            # Transform CV text using fitted TF-IDF vectorizer
            cv_vector = self.vectorizer.transform([cv_text])
            
            # Find K nearest neighbors
            distances, indices = self.knn_model.kneighbors(cv_vector)
            
            # Convert distances to similarity scores (1 - distance)
            similarities = 1 - distances[0]
            
            # Prepare recommendations
            recommendations = []
            for idx, similarity in zip(indices[0], similarities):
                job_id = self.job_ids[idx]
                
                # Skip jobs with too low similarity
                if similarity < self.match_threshold:
                    continue
                
                # Get job data
                job_data = self.job_data.get(job_id, {})
                if not job_data:
                    continue
                
                # Create recommendation entry
                recommendation = {
                    'job_id': job_id,
                    'job_title': job_data.get('job_title', ''),
                    'match_score': float(similarity),
                    'content_similarity': float(similarity),
                    'employer_id': job_data.get('employer_id', 0),
                    'category_id': job_data.get('category_id', 0),
                    'level_id': job_data.get('level_id', 0),
                    'reason': f"Công việc này có nội dung phù hợp với hồ sơ của bạn (độ tương đồng: {similarity:.2f})"
                }
                
                recommendations.append(recommendation)
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in KNN recommendation: {e}")
            return []