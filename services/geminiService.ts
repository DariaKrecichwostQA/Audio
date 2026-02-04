
import { GoogleGenAI } from "@google/genai";

export const geminiService = {
  async analyzeAnomalies(anomalies: any[]) {
    // API key must be obtained exclusively from the environment variable process.env.API_KEY.
    if (!process.env.API_KEY) return "Brak klucza API dla analizy Gemini.";
    
    try {
      // Create a new GoogleGenAI instance right before making an API call to ensure it always uses the most up-to-date API key.
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      // Use gemini-3-pro-preview for machine diagnostic reasoning which is a complex reasoning task.
      const response = await ai.models.generateContent({
        model: 'gemini-3-pro-preview',
        contents: `Analiza zdarzeń akustycznych maszyny: ${JSON.stringify(anomalies.map(a => ({
          time: a.offsetSeconds,
          intensity: a.intensity,
          severity: a.severity
        })))}. Opisz potencjalne przyczyny techniczne (łożyska, kawitacja, luzy) i zaproponuj kroki serwisowe w 3 punktach.`,
        config: {
          systemInstruction: "Jesteś ekspertem utrzymania ruchu i diagnostyki wibroakustycznej. Odpowiadaj konkretnie po polsku.",
          // Enable thinking for better technical diagnostic quality
          thinkingConfig: { thinkingBudget: 2048 }
        }
      });
      // Extract text via the .text property (not a method).
      return response.text || "Błąd generowania opinii.";
    } catch (err) {
      console.error("Gemini analysis error:", err);
      return "Błąd połączenia z Gemini AI.";
    }
  }
};
