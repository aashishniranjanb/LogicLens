import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util

class AtomicAnswerValidator:
    def __init__(self):
        print("Initializing Validator Pipeline...")
        print(" [1/2] Loading SBERT Retriever (all-MiniLM-L6-v2)...")
        self.retriever = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(" [2/2] Loading NLI Verifier (distilroberta-base)...")
        # We use a Cross-Encoder for high-precision logic checking
        self.verifier = CrossEncoder('cross-encoder/nli-distilroberta-base')
        
        # Mapping model output labels to readable verdicts
        self.label_map = ['contradiction', 'entailment', 'neutral']
        print("System Ready.\n")

    def decompose_claims(self, text):
        """
        Splits a complex answer into atomic factual claims.
        In production, this would use a generative LLM. Here we use a robust delimiter split.
        """
        # Split by periods/semicolons and filter empty strings
        raw_claims = text.replace(';', '.').split('.')
        claims = [c.strip() for c in raw_claims if len(c.strip()) > 10]
        return claims

    def get_best_evidence(self, claim, context_sentences):
        """
        Retrieves the single best sentence from context that matches the claim.
        """
        claim_emb = self.retriever.encode(claim, convert_to_tensor=True)
        context_embs = self.retriever.encode(context_sentences, convert_to_tensor=True)
        
        # Find highest cosine similarity
        scores = util.cos_sim(claim_emb, context_embs)[0]
        best_idx = torch.argmax(scores).item()
        return context_sentences[best_idx], scores[best_idx].item()

    def verify_logic(self, claim, evidence_text, threshold=0.35):
        """
        Uses NLI to check if Evidence logically entails the Claim.
        """
        # 1. Hallucination Check: If no relevant evidence exists in context
        if self.get_similarity(claim, evidence_text) < threshold:
             return "Not Supported (Low Relevance)", 0.0

        # 2. Logic Check: Run (Premise, Hypothesis) through CrossEncoder
        scores = self.verifier.predict([(evidence_text, claim)])
        probs = torch.nn.functional.softmax(torch.tensor(scores), dim=1).numpy()[0]
        
        pred_idx = np.argmax(probs)
        verdict = self.label_map[pred_idx]
        return verdict, probs[pred_idx]

    def get_similarity(self, text1, text2):
        # Helper for quick similarity check
        e1 = self.retriever.encode(text1, convert_to_tensor=True)
        e2 = self.retriever.encode(text2, convert_to_tensor=True)
        return util.cos_sim(e1, e2).item()

    def validate(self, question, context, machine_answer):
        # Preprocess context into sentences for retrieval
        context_sentences = [s.strip() for s in context.split('.') if len(s) > 5]
        claims = self.decompose_claims(machine_answer)
        
        report = []
        is_fully_correct = True
        
        for claim in claims:
            evidence, sim_score = self.get_best_evidence(claim, context_sentences)
            verdict, conf = self.verify_logic(claim, evidence)
            
            # Aggregation Rule: Any contradiction or neutral finding invalidates the answer
            status = "✅ Correct"
            if verdict in ['contradiction', 'neutral', 'Not Supported (Low Relevance)']:
                status = "❌ Incorrect"
                is_fully_correct = False
            
            report.append({
                "claim": claim,
                "evidence_found": evidence,
                "logic_verdict": verdict,
                "status": status,
                "confidence": round(float(conf), 4)
            })

        final_label = "Correct" if is_fully_correct else "Incorrect"
        
        return {
            "final_decision": final_label,
            "detailed_report": report
        }
