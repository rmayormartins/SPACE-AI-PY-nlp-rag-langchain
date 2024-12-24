import os
from typing import List, Tuple, Dict
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import gradio as gr
import torch

class EnhancedRAGSystem:
    def __init__(self):
        try:
            print("Initializing RAG System...")
            self.chunk_size = 500
            self.chunk_overlap = 50
            self.k_documents = 4
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len
            )
            
            print("Loading embedding model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            print("Loading language model...")
            self.llm_model_name = "google/flan-t5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model_name)
            
            self.prompt_template = PromptTemplate(
                template="""Use the context below to answer the question. 
                If the answer is not in the context, say "I don't have enough information in the context to answer this question."
                
                Context: {context}
                Question: {question}
                
                Detailed answer:""",
                input_variables=["context", "question"]
            )
            
            print("Setting up pipeline...")
            self.pipe = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                device=-1,
                model_kwargs={"temperature": 0.7}
            )
            
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            print("RAG System initialized successfully!")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def process_documents(self, text: str) -> bool:
        try:
            print("Processing documents...")
            if not text or len(text.strip()) < 10:
                print("Text is too short or empty")
                return False
                
            print("Splitting text...")
            texts = self.text_splitter.split_text(text)
            
            print("Creating vectorstore...")
            self.vectorstore = Chroma.from_texts(
                texts,
                self.embeddings,
                metadatas=[{"source": f"chunk_{i}", "text": t} for i, t in enumerate(texts)]
            )
            
            print("Setting up retriever...")
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.k_documents}
            )
            
            print("Creating QA chain...")
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt_template}
            )
            
            print("Documents processed successfully!")
            return True
            
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            return False

    def answer_question(self, question: str) -> Tuple[str, str]:
        try:
            print(f"Answering question: {question}")
            if not hasattr(self, 'qa_chain'):
                return "Please process some documents first.", ""
                
            response = self.qa_chain({"query": question})
            answer = response["result"]
            
            sources = []
            for i, doc in enumerate(response["source_documents"], 1):
                text_preview = doc.page_content[:100] + "..."
                sources.append(f"Excerpt {i}: {text_preview}")
            
            sources_text = "\n".join(sources)
            print("Answer generated successfully!")
            return answer, sources_text
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}", ""

def create_enhanced_interface():
    try:
        print("Creating interface...")
        rag_system = EnhancedRAGSystem()
        
        def process_and_answer(text: str, question: str) -> str:
            print("Processing new request...")
            if not text.strip() or not question.strip():
                return "Please provide both text and question."
            
            success = rag_system.process_documents(text)
            if not success:
                return "Error processing the text. Please check if the text is valid and try again."
            
            answer, sources = rag_system.answer_question(question)
            
            if sources:
                return f"""Answer: {answer}

Relevant excerpts consulted:
{sources}"""
            return answer

        custom_css = """
            .custom-description {
                margin-bottom: 20px;
                text-align: center;
            }
            .custom-description a {
                text-decoration: none;
                color: #007bff;
                margin: 0 5px;
            }
            .custom-description a:hover {
                text-decoration: underline;
            }
        """

        with gr.Blocks(css=custom_css) as interface:
            gr.HTML("""
                <div class="custom-description">
                    <h1>Advanced RAG with Multilingual Support</h1>
                    <p>Ramon Mayor Martins: 
                        <a href="https://rmayormartins.github.io/" target="_blank">Website</a> | 
                        <a href="https://huggingface.co/rmayormartins" target="_blank">Spaces</a> |
                        <a href="https://github.com/rmayormartins" target="_blank">GitHub</a>
                    </p>
                    <p>This system uses Retrieval-Augmented Generation (RAG) to answer questions about your texts in multiple languages.
                    Simply paste your text and ask questions in any language!</p>
                </div>
            """)
            
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Base Text",
                        placeholder="Paste here the text that will serve as knowledge base...",
                        lines=10
                    )
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="What would you like to know about the text?"
                    )
                    submit_btn = gr.Button("Submit")
                
                with gr.Column():
                    output = gr.Textbox(label="Answer")

            examples = [
                # English example
                ["The solar system consists of the Sun and the celestial bodies that orbit it. These include eight planets (Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune), their moons, asteroids, comets, and other objects.",
                 "How many planets are in the solar system?"],
                
                # Spanish example
                ["El sistema solar está formado por el Sol y los cuerpos celestes que orbitan a su alrededor. Estos incluyen ocho planetas (Mercurio, Venus, Tierra, Marte, Júpiter, Saturno, Urano y Neptuno), sus lunas, asteroides, cometas y otros objetos.",
                 "¿Cuántos planetas hay en el sistema solar?"],
                
                # Portuguese example
                ["O sistema solar é composto pelo Sol e pelos corpos celestes que orbitam ao seu redor. Isso inclui oito planetas (Mercúrio, Vênus, Terra, Marte, Júpiter, Saturno, Urano e Netuno), suas luas, asteroides, cometas e outros objetos.",
                 "Quantos planetas existem no sistema solar?"]
            ]

            gr.Examples(
                examples=examples,
                inputs=[text_input, question_input],
                outputs=output,
                fn=process_and_answer,
                cache_examples=True
            )

            submit_btn.click(
                fn=process_and_answer,
                inputs=[text_input, question_input],
                outputs=output
            )

        print("Interface created successfully!")
        return interface
        
    except Exception as e:
        print(f"Error creating interface: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting application...")
    try:
        demo = create_enhanced_interface()
        demo.launch()
    except Exception as e:
        print(f"Application failed to start: {str(e)}")