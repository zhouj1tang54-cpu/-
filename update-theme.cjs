const fs = require('fs');

const files = [
  'components/LiveTutor.tsx',
  'components/Transcript.tsx',
  'components/AudioVisualizer.tsx'
];

const replacements = [
  { from: /(?<!dark:)\bbg-gray-950\b/g, to: 'bg-gray-50 dark:bg-gray-950' },
  { from: /(?<!dark:)\bbg-gray-900\b/g, to: 'bg-white dark:bg-gray-900' },
  { from: /(?<!dark:)\bbg-gray-800\b/g, to: 'bg-gray-100 dark:bg-gray-800' },
  { from: /(?<!dark:)\bbg-gray-750\b/g, to: 'bg-gray-50 dark:bg-gray-750' },
  { from: /(?<!dark:)\bbg-gray-700\b/g, to: 'bg-gray-200 dark:bg-gray-700' },
  { from: /(?<!dark:)\bbg-gray-600\b/g, to: 'bg-gray-300 dark:bg-gray-600' },
  
  { from: /(?<!dark:)\btext-white\b/g, to: 'text-gray-900 dark:text-white' },
  { from: /(?<!dark:)\btext-gray-100\b/g, to: 'text-gray-800 dark:text-gray-100' },
  { from: /(?<!dark:)\btext-gray-200\b/g, to: 'text-gray-700 dark:text-gray-200' },
  { from: /(?<!dark:)\btext-gray-300\b/g, to: 'text-gray-600 dark:text-gray-300' },
  { from: /(?<!dark:)\btext-gray-400\b/g, to: 'text-gray-500 dark:text-gray-400' },
  { from: /(?<!dark:)\btext-gray-500\b/g, to: 'text-gray-400 dark:text-gray-500' },
  
  { from: /(?<!dark:)\bborder-gray-800\b/g, to: 'border-gray-200 dark:border-gray-800' },
  { from: /(?<!dark:)\bborder-gray-700\b/g, to: 'border-gray-300 dark:border-gray-700' },
  { from: /(?<!dark:)\bborder-gray-600\b/g, to: 'border-gray-400 dark:border-gray-600' },
  
  { from: /(?<!dark:)\bring-gray-800\b/g, to: 'ring-gray-200 dark:ring-gray-800' },
];

files.forEach(file => {
  if (!fs.existsSync(file)) {
    console.log(`File not found: ${file}`);
    return;
  }
  let content = fs.readFileSync(file, 'utf-8');
  replacements.forEach(r => {
    content = content.replace(r.from, r.to);
  });
  fs.writeFileSync(file, content);
  console.log(`Updated ${file}`);
});
