import io
import time
import logging
import json
import re
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = "AQ.Ab8RN6L5wiWFrYQrs4aH2boishQU3BOsec4abz3zfCyCjfHcyA"
client = genai.Client(api_key=API_KEY)

app = FastAPI(title="Bubble Sheet OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROMPT = """You are an expert OMR bubble sheet scanner for IUB (Islamia University of Bahawalpur) Examinations Department.

This is an actual scanned bubble sheet image. Analyze it with EXTREME care and full attention.

BUBBLE SHEET STRUCTURE:
- Questions are numbered 1 to 60
- Each question has 5 options: A, B, C, D, E
- Filled/darkened bubble (solid circle) = selected answer
- Empty circle = not selected
- Questions are arranged in 3 columns: Q1-Q20 (left), Q21-Q40 (middle), Q41-Q60 (right)

YOUR TASK:
Step 1 - Read student info from top section of sheet carefully
Step 2 - For each question 1 to 60, identify which bubble is filled/darkened
Step 3 - Return everything in exact JSON format below

Return ONLY a valid JSON object, no markdown, no backticks, no explanation, no extra text:
{
  "applicant_name": "full name here or N/A",
  "father_name": "father name here or N/A",
  "roll_number": "roll number here or N/A",
  "subject_name": "subject and semester here or N/A",
  "booklet_type": "A or B or C or D or N/A",
  "answers": {
    "1": "B",
    "2": "E",
    "3": "A",
    "4": "C",
    "5": "D",
    "6": "B",
    "7": "A",
    "8": "B",
    "9": "D",
    "10": "E",
    "11": "A",
    "12": "D",
    "13": "E",
    "14": "A",
    "15": "A",
    "16": "E",
    "17": "E",
    "18": "B",
    "19": "D",
    "20": "C",
    "21": "C",
    "22": "E",
    "23": "A",
    "24": "B",
    "25": "B",
    "26": "B",
    "27": "C",
    "28": "D",
    "29": "D",
    "30": "A",
    "31": "B",
    "32": "B",
    "33": "E",
    "34": "A",
    "35": "D",
    "36": "A",
    "37": "B",
    "38": "C",
    "39": "B",
    "40": "E",
    "41": "D",
    "42": "B",
    "43": "B",
    "44": "B",
    "45": "E",
    "46": "D",
    "47": "C",
    "48": "A",
    "49": "A",
    "50": "C",
    "51": "B",
    "52": "B",
    "53": "D",
    "54": "A",
    "55": "C",
    "56": "E",
    "57": "D",
    "58": "B",
    "59": "B",
    "60": "D"
  }
}

STRICT RULES:
- answers mein EXACTLY 1 se 60 tak sab questions hone chahiye, koi miss nahi hona
- Valid options sirf: A, B, C, D, or E
- Agar koi bubble clearly marked nahi ho toh: "X"
- ONLY pure JSON return karna, koi bhi extra text, explanation ya markdown nahi
- JSON structure bilkul same rakhna jaise upar diya gaya hai
"""

@app.post("/process-bubble-sheet/")
async def process_bubble_sheet(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        logger.error(f"Image read error: {str(e)}")
        raise HTTPException(status_code=500, detail="Image read karne mein masla hua.")

    max_retries = 3
    retry_delay = 2
    raw_text = ""

    for attempt in range(max_retries):
        try:
            logger.info(f"Processing image with New SDK: Attempt {attempt + 1}/{max_retries}")

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[pil_image, PROMPT],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                )
            )

            raw_text = response.text.strip()
            logger.info(f"Raw Gemini Response (first 500 chars): {raw_text[:500]}")

            # Markdown backticks remove karo agar hon
            raw_text = re.sub(r"```json|```", "", raw_text).strip()

            parsed = json.loads(raw_text)

            # Validate ke answers mein data hai
            answers_dict = parsed.get("answers", {})
            if len(answers_dict) == 0:
                raise ValueError("Gemini ne koi answers return nahi kiye")

            # Answers ko array format mein convert karo HTML ke liye
            answers_array = []
            for q_num, option in answers_dict.items():
                answers_array.append({
                    "question_number": int(q_num),
                    "selected_option": option.upper() if option else "X"
                })

            # Question number ke hisaab se sort karo
            answers_array.sort(key=lambda x: x["question_number"])

            logger.info(f"Successfully extracted {len(answers_array)} answers")

            return {
                "status": "success",
                "data": {
                    "applicant_name": parsed.get("applicant_name", "N/A"),
                    "father_name": parsed.get("father_name", "N/A"),
                    "roll_number": parsed.get("roll_number", "N/A"),
                    "subject_name": parsed.get("subject_name", "N/A"),
                    "booklet_type": parsed.get("booklet_type", "N/A"),
                    "answers": answers_array
                }
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error on attempt {attempt + 1}: {str(e)}")
            logger.error(f"Raw text was: {raw_text[:500]}")
            if attempt < max_retries - 1:
                logger.warning(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            raise HTTPException(
                status_code=500,
                detail=f"Gemini ne valid JSON nahi diya. Error: {str(e)}"
            )

        except ValueError as e:
            logger.error(f"Validation Error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            raise HTTPException(
                status_code=500,
                detail=f"Data validation failed: {str(e)}"
            )

        except Exception as e:
            error_msg = str(e).upper()
            logger.warning(f"Attempt {attempt + 1} failed. Error: {str(e)}")

            if "503" in error_msg or "UNAVAILABLE" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if attempt < max_retries - 1:
                    logger.warning(f"Server busy. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue

            raise HTTPException(
                status_code=500,
                detail=f"API Request Failed: {str(e)}"
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)