import React from 'react';

function WorkspaceTree({ navHistory, currentNodes, handleNodeClick, handleBack }) {
  return (
    <div>
      {navHistory.length > 0 && (
        <div style={{ marginBottom: '10px' }}>
          <button onClick={handleBack}>Back</button>
          <div style={{ marginTop: '5px' }}>
            <strong>Current Path: </strong>
            {navHistory.map((node, index) => (
              <span key={node.name}>
                {node.name}{index < navHistory.length - 1 && ' > '}
              </span>
            ))}
          </div>
        </div>
      )}
      {currentNodes.length > 0 && (
        <ul>
          {currentNodes.map(node => (
            <li
              key={node.name}
              style={{ cursor: 'pointer', padding: '4px 0' }}
              onClick={() => handleNodeClick(node)}
            >
              {node.name}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default WorkspaceTree;
