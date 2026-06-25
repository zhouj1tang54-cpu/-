import { GoogleGenAI } from '@google/genai';

async function test() {
  const envKey = process.env.GEMINI_API_KEY || '';
  const fallbackKey = 'AIzaSyBaGTea1IFQrkKHfyEiQ3ZTmCXBPj7VoPA';
  const apiKey = envKey && envKey !== 'MY_GEMINI_API_KEY' ? envKey : fallbackKey;
  
  console.log('Testing key prefix:', apiKey.substring(0, 8));
  
  const ai = new GoogleGenAI({ apiKey });
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: 'Say Hello',
    });
    console.log('Success! Response:', response.text);
  } catch (err) {
    console.error('Failed to generate content:', err);
  }
}

test();
