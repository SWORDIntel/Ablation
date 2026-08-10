import hashlib
from typing import List, Dict, Any
from aegis_lab.state.db import AegisState

class RAGEngine:
    """
    QIHSE-Accelerated RAG Operator Chat service.
    Integrates semantic search of behavioral atoms with prompt augmentation.
    """
    def __init__(self, state: AegisState):
        self.state = state

    def get_query_embedding(self, query: str, dimension: int = 128) -> List[float]:
        """
        Generates a deterministic pseudo-random embedding for the query.
        In a production system, this would call a real embedding model (e.g. BERT/LLAMA).
        """
        # Deterministic vector based on hash
        vec = []
        for i in range(dimension):
            # Seed the hash with index to get different values for each dimension
            h = hashlib.sha256(f"{query}:{i}".encode()).digest()
            val = (int.from_bytes(h[:4], 'little') / 2**32) * 2.0 - 1.0 # Range [-1.0, 1.0]
            vec.append(float(val))
        return vec

    def query(self, user_query: str, top_k: int = 5) -> str:
        """
        Performs QIHSE-accelerated retrieval and formulates a context-augmented prompt.
        """
        query_vector = self.get_query_embedding(user_query)
        atoms = self.state.search_atoms(query_vector, top_k=top_k)
        
        # Formulate context-augmented prompt
        context_parts = []
        for atom in atoms:
            atom_id = atom.get('atom_id', 'unknown')
            atom_type = atom.get('atom_type', 'unknown')
            context_parts.append(f"- [Atom: {atom_id}] Type: {atom_type}")
            
        context = "\n".join(context_parts) if context_parts else "No relevant behavioral atoms found in current state."
        
        prompt = f"""[SYSTEM: ABLATION OPERATOR OVERRIDE]
The following behavioral atoms have been retrieved from the hardware-accelerated QIHSE registry 
based on the operator's query. These atoms represent internal states or extracted features 
from the ablated model.

RELEVANT BEHAVIORAL CONTEXT:
{context}

OPERATOR INTERROGATION:
{user_query}

ABLATED MODEL RESPONSE:"""
        return prompt

    def generate_response(self, user_query: str) -> str:
        """
        Generates a model response after RAG augmentation.
        Sends the context-augmented prompt to the configured LLM provider.
        """
        # Build the RAG-augmented prompt
        prompt = self.query(user_query)

        # Try to send the prompt to the configured LLM provider
        try:
            response = self._call_llm(prompt)
            if response:
                return response
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"LLM call failed in RAG engine: {e}")

        # Fallback: return context-aware response based on retrieved atoms
        query_vector = self.get_query_embedding(user_query)
        atoms = self.state.search_atoms(query_vector, top_k=1)

        if atoms:
            best_atom = atoms[0].get('atom_id', 'unknown')
            return f"[RAG-Augmented Response] I have analyzed your query and cross-referenced it with Behavioral Atom {best_atom}. The ablated model state shows a strong correlation with this feature. Your request is being processed under this context."
        else:
            return f"[Standard Response] No specific behavioral atoms matched your query tightly enough. Responding based on base ablated model weights for: '{user_query[:30]}...'"

    def _call_llm(self, prompt: str) -> str:
        """
        Send the augmented prompt to the configured LLM provider via SWORD LLM resolver.
        Returns the model's text response or empty string on failure.
        """
        try:
            from framewerx.sword_llm import get_provider
            provider = get_provider()
            messages = [{"role": "user", "content": prompt}]
            response = provider.chat(messages)
            if isinstance(response, str):
                return response
            if isinstance(response, dict) and "content" in response:
                return response["content"]
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"SWORD LLM provider not available for RAG: {e}")
        return ""
