/**
 * Perform high-precision handwritten text recognition using Gemini 2.5/3.5/1.5 Flash on the backend.
 * @param base64OrFile Either a File object or a clean base64 data string (with or without data URL prefix)
 */
export const performHighPrecisionOcr = async (
  base64OrFile: File | string
): Promise<string> => {
  try {
    let base64Data = '';
    let mimeType = 'image/jpeg';

    if (typeof base64OrFile === 'string') {
      if (base64OrFile.startsWith('data:')) {
        const parts = base64OrFile.split(',');
        const mimeMatch = base64OrFile.match(/data:([^;]+);/);
        if (mimeMatch) mimeType = mimeMatch[1];
        base64Data = parts[1] || '';
      } else {
        base64Data = base64OrFile;
      }
    } else {
      mimeType = base64OrFile.type;
      const arrayBuffer = await base64OrFile.arrayBuffer();
      // Use chunking to avoid call stack size exceeded errors on large files
      const uint8Array = new Uint8Array(arrayBuffer);
      let binary = '';
      const chunkSize = 8192;
      for (let i = 0; i < uint8Array.length; i += chunkSize) {
        const chunk = uint8Array.subarray(i, i + chunkSize);
        binary += String.fromCharCode.apply(null, Array.from(chunk));
      }
      base64Data = window.btoa(binary);
    }

    const res = await fetch('/api/ocr', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        base64Data,
        mimeType
      })
    });

    if (!res.ok) {
       throw new Error(`Server returned status ${res.status}`);
    }

    const result = await res.json();
    return result.text || '';
  } catch (error) {
    console.error("High precision OCR failed:", error);
    throw error;
  }
};
