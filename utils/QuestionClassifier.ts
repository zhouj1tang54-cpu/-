import { GoogleGenAI, Type } from '@google/genai';

export interface QuestionClassification {
  subject: string;
  topic: string;
  questionType: string;
  difficulty: 'Easy' | 'Medium' | 'Hard' | 'Unknown';
  keyConcepts: string[];
}

export const classifyQuestion = async (
  text: string,
  apiKey: string
): Promise<QuestionClassification | null> => {
  if (!text || !apiKey) return null;

  try {
    const ai = new GoogleGenAI({ apiKey });
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: `请分析以下题目内容，并判断它的学科、知识点、题型、难度以及核心概念。\n\n题目内容：\n${text}`,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            subject: { type: Type.STRING, description: "学科，例如：数学、物理、化学、英语、语文等" },
            topic: { type: Type.STRING, description: "具体知识点，例如：二次函数、牛顿第二定律、阅读理解等" },
            questionType: { type: Type.STRING, description: "题型，例如：选择题、填空题、解答题、证明题、作文等" },
            difficulty: { type: Type.STRING, description: "难度评估，必须是 Easy, Medium, Hard, Unknown 之一" },
            keyConcepts: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "核心概念列表"
            }
          },
          required: ["subject", "topic", "questionType", "difficulty", "keyConcepts"]
        }
      }
    });

    if (response.text) {
      return JSON.parse(response.text) as QuestionClassification;
    }
    return null;
  } catch (error) {
    console.error("Error classifying question:", error);
    return null;
  }
};
