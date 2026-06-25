export interface QuestionClassification {
  subject: string;
  topic: string;
  questionType: string;
  difficulty: 'Easy' | 'Medium' | 'Hard' | 'Unknown';
  keyConcepts: string[];
}

export const classifyQuestion = async (
  text: string,
  _apiKey?: string
): Promise<QuestionClassification | null> => {
  if (!text) return null;

  try {
    const res = await fetch('/api/classify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text })
    });

    if (!res.ok) {
       throw new Error(`Server returned status ${res.status}`);
    }

    const data = await res.json();
    const parsedText = data.text ? data.text.trim() : '';

    if (parsedText) {
      let cleanedText = parsedText;
      if (cleanedText.startsWith('```json')) {
        cleanedText = cleanedText.substring(7);
      }
      if (cleanedText.endsWith('```')) {
        cleanedText = cleanedText.substring(0, cleanedText.length - 3);
      }
      cleanedText = cleanedText.trim();

      try {
        return JSON.parse(cleanedText) as QuestionClassification;
      } catch (jsonErr) {
        console.error("Failed to parse classification JSON:", cleanedText, jsonErr);
      }
    }
    return {
      subject: "数学",
      topic: "核心解题分析",
      questionType: "解答题",
      difficulty: "Medium",
      keyConcepts: ["核心解题思维"]
    };
  } catch (error) {
    console.error("Error classifying question:", error);
    return null;
  }
};
