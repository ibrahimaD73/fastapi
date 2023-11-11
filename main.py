
from uuid import uuid4

import openai 

import PyPDF2
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from os import path

from schema import SummarizeRequestModel, ExtractSkillsRequestModel, MatchJobCandidateRequestModel

class APIServer:
    def __init__(self, host:str, port:int, openai_api_key:str):
        self.host = host
        self.port = port
        self.openai_api_key = openai_api_key
        
        openai.api_key = self.openai_api_key

        self.app = FastAPI(
            title="API Server for skin reaction detection",
            description="""
                API Server for skin reaction detection based on the model trained by the team.
            """,
            version="0.1.0",
        )
    
        self.app.add_api_route(path="/heartbeat", endpoint=self.heartbeat_handler, methods=["GET"])
        self.app.add_api_route(path="/upload_pdf", endpoint=self.upload_pdf_handler, methods=["POST"])
        self.app.add_api_route(path="/extract_text", endpoint=self.extract_text_handler, methods=["POST"])
        self.app.add_api_route(path="/summarize_text", endpoint=self.summarize_text_handler, methods=["POST"])
        self.app.add_api_route(path="/extract_skills", endpoint=self.extract_skills, methods=["POST"])
        self.app.add_api_route(path="/match_job_candidate", endpoint=self.match_job_candidate, methods=["POST"])

    def run(self):
        uvicorn.run(app=self.app, host=self.host, port=self.port)

    async def heartbeat_handler(self):
        return JSONResponse(
            status_code=200,
            content={
                "status": "running",
                "message": "API Server is running"
            }
        )

    async def upload_pdf_handler(self, file: UploadFile = File(...)):
        file_id = uuid4()
        binarystream = await file.read()
        
        with open(f"volume/{file_id}.pdf", "wb") as file_pointer:
            file_pointer.write(binarystream)

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "file_id": str(file_id),
                "message": "File uploaded successfully"
            }
        )

    async def extract_text_handler(self, file_id:str, page_number:int):
        path2pdf_file = f"volume/{file_id}.pdf"

        if not path.exists(path2pdf_file):
            raise HTTPException(status_code=404, detail="File not found")
        
        reader = PyPDF2.PdfReader(stream=path2pdf_file)
        pages = reader.pages # list of pages
        page = pages[page_number] # get the page

        return JSONResponse(
            status_code=200,
            content={
                "extracted_text": page.extract_text() 
            } 
        )

    # Mettez à jour les fonctions pour utiliser la nouvelle interface de OpenAI

    async def summarize_text_handler(self, summarize_req: SummarizeRequestModel):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Tu es un service qui permet de resumer des textes"},
                {"role": "user", "content": summarize_req.text},
            ],
        )
        return JSONResponse(
            status_code=200,
            content={"summary": response["choices"][0]["message"]["content"]},
        )

    async def extract_skills(self, extract_skills_req: ExtractSkillsRequestModel):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Tu es un service qui permet d'extraire des skills à partir d'un CV. Les skills doivent être séparés par des virgules."},
                {"role": "user", "content": extract_skills_req.text},
            ],
        )
        return JSONResponse(
            status_code=200,
            content={"summary": response["choices"][0]["message"]["content"]},
        )

    async def match_job_candidate(self, match_jon_candidate: MatchJobCandidateRequestModel):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Tu es un service qui permet de calculer un score de pertinence entre la description d'une offre d'emploi et d'un CV. Le score doit être compris entre 0 et 1. Tu dois justifier le choix du score."},
                {"role": "user", "content": f"""
                    CV : {match_jon_candidate.candidate_resume}
                    Offre d'emploi : {match_jon_candidate.job_description}
                """},
            ],
        )
        return JSONResponse(
            status_code=200,
            content={"summary": response["choices"][0]["message"]["content"]},
        )



if __name__ == "__main__":
    server = APIServer(host="0.0.0.0", port=8000, openai_api_key='your_api_key')
    server.run()  
