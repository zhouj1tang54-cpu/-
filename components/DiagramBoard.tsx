import React from 'react';
import { X, Maximize2, Minimize2 } from 'lucide-react';

export interface Shape {
  id: string;
  type: 'line' | 'circle' | 'rect' | 'text' | 'polygon' | 'arrow';
  x?: number;
  y?: number;
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
  r?: number;
  width?: number;
  height?: number;
  points?: string; // "x1,y1 x2,y2 ..."
  content?: string;
  color?: string;
  strokeWidth?: number;
  fill?: string;
  label?: string;
}

export interface DiagramData {
  title: string;
  description?: string;
  shapes: Shape[];
  viewBox?: string; // "0 0 100 100"
}

interface DiagramBoardProps {
  data: DiagramData | null;
  onClose: () => void;
}

const DiagramBoard: React.FC<DiagramBoardProps> = ({ data, onClose }) => {
  const [isExpanded, setIsExpanded] = React.useState(false);

  if (!data) return null;

  const defaultViewBox = "0 0 400 300";
  const viewBox = data.viewBox || defaultViewBox;

  return (
    <div 
      className={`absolute transition-all duration-300 ease-in-out bg-gray-900/95 backdrop-blur-md border border-indigo-500/30 shadow-2xl rounded-2xl overflow-hidden z-50 flex flex-col ${
        isExpanded 
          ? 'inset-4' 
          : 'bottom-32 right-4 w-80 h-64'
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-3 bg-gray-800/50 border-b border-gray-700">
        <h3 className="text-indigo-300 font-semibold text-sm truncate flex-1">
          {data.title || "知识点图示"}
        </h3>
        <div className="flex items-center gap-1">
          <button 
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
          >
            {isExpanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
          <button 
            onClick={onClose}
            className="p-1.5 text-gray-400 hover:text-red-400 hover:bg-red-900/20 rounded-lg transition-colors"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4 flex items-center justify-center bg-gray-900 relative">
        {/* Grid Background */}
        <div className="absolute inset-0 opacity-10 pointer-events-none" 
             style={{ 
               backgroundImage: 'radial-gradient(#6366f1 1px, transparent 1px)', 
               backgroundSize: '20px 20px' 
             }} 
        />

        <svg 
          viewBox={viewBox} 
          className="w-full h-full max-w-full max-h-full"
          preserveAspectRatio="xMidYMid meet"
        >
          {data.shapes.map((shape) => {
            const stroke = shape.color || "#e0e7ff";
            const fill = shape.fill || "none";
            const strokeWidth = shape.strokeWidth || 2;

            switch (shape.type) {
              case 'line':
                return (
                  <g key={shape.id}>
                    <line 
                      x1={shape.x1} y1={shape.y1} 
                      x2={shape.x2} y2={shape.y2} 
                      stroke={stroke} 
                      strokeWidth={strokeWidth} 
                      strokeLinecap="round"
                    />
                    {shape.label && (
                      <text 
                        x={(shape.x1! + shape.x2!) / 2} 
                        y={(shape.y1! + shape.y2!) / 2 - 5} 
                        fill={stroke} 
                        fontSize="12" 
                        textAnchor="middle"
                      >
                        {shape.label}
                      </text>
                    )}
                  </g>
                );
              case 'arrow':
                return (
                  <g key={shape.id}>
                    <defs>
                      <marker id={`arrowhead-${shape.id}`} markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill={stroke} />
                      </marker>
                    </defs>
                    <line 
                      x1={shape.x1} y1={shape.y1} 
                      x2={shape.x2} y2={shape.y2} 
                      stroke={stroke} 
                      strokeWidth={strokeWidth} 
                      markerEnd={`url(#arrowhead-${shape.id})`}
                    />
                    {shape.label && (
                      <text 
                        x={(shape.x1! + shape.x2!) / 2} 
                        y={(shape.y1! + shape.y2!) / 2 - 5} 
                        fill={stroke} 
                        fontSize="12" 
                        textAnchor="middle"
                      >
                        {shape.label}
                      </text>
                    )}
                  </g>
                );
              case 'circle':
                return (
                  <g key={shape.id}>
                    <circle 
                      cx={shape.x} cy={shape.y} r={shape.r} 
                      stroke={stroke} 
                      strokeWidth={strokeWidth} 
                      fill={fill} 
                    />
                    {shape.label && (
                      <text 
                        x={shape.x} y={shape.y! + shape.r! + 15} 
                        fill={stroke} 
                        fontSize="12" 
                        textAnchor="middle"
                      >
                        {shape.label}
                      </text>
                    )}
                  </g>
                );
              case 'rect':
                return (
                  <g key={shape.id}>
                    <rect 
                      x={shape.x} y={shape.y} 
                      width={shape.width} height={shape.height} 
                      stroke={stroke} 
                      strokeWidth={strokeWidth} 
                      fill={fill} 
                      rx="4"
                    />
                    {shape.label && (
                      <text 
                        x={shape.x! + shape.width! / 2} 
                        y={shape.y! + shape.height! / 2} 
                        fill={stroke} 
                        fontSize="12" 
                        textAnchor="middle" 
                        alignmentBaseline="middle"
                      >
                        {shape.label}
                      </text>
                    )}
                  </g>
                );
              case 'polygon':
                return (
                  <g key={shape.id}>
                    <polygon 
                      points={shape.points} 
                      stroke={stroke} 
                      strokeWidth={strokeWidth} 
                      fill={fill} 
                    />
                    {shape.label && (
                      // Simple label positioning (approximate center)
                      <text 
                        x={10} y={10} // Placeholder, hard to calculate center without parsing points
                        fill={stroke} 
                        fontSize="12" 
                      >
                        {shape.label}
                      </text>
                    )}
                  </g>
                );
              case 'text':
                return (
                  <text 
                    key={shape.id}
                    x={shape.x} y={shape.y} 
                    fill={stroke} 
                    fontSize={shape.height || 14} 
                    textAnchor="middle"
                  >
                    {shape.content}
                  </text>
                );
              default:
                return null;
            }
          })}
        </svg>
      </div>
      
      {data.description && (
        <div className="p-3 bg-gray-800/50 border-t border-gray-700 text-xs text-gray-300">
          {data.description}
        </div>
      )}
    </div>
  );
};

export default DiagramBoard;
