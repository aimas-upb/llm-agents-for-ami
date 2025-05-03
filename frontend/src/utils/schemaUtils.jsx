export const getHumanReadableProperty = (uri) => {
    const parts = uri.split(/[#\/]/);
    const propertyName = parts[parts.length - 1];
    
    const propertyMap = {
      // TD properties
      'title': 'Title',
      'hasSecurityConfiguration': 'Security Configuration',
      'hasActionAffordance': 'Action',
      'hasEventAffordance': 'Event',
      'hasPropertyAffordance': 'Property',
      
      // HMAS properties
      'isContainedIn': 'Container',
      'contains': 'Contains'
    };
    
    return propertyMap[propertyName] || propertyName;
  };
  
  export const getHumanReadableValue = (value) => {
    if (!value) return '';
    
    if (value.startsWith('http://') || value.startsWith('https://')) {
      if (value.includes('/workspaces/')) {
        const match = value.match(/\/workspaces\/([^/#]+)/);
        if (match && match[1]) {
          return match[1];
        }
      }
      
      const parts = value.split(/[#\/]/);
      const lastPart = parts[parts.length - 1];
      
      const valueMap = {
        'NoSecurityScheme': 'No Security',
        'artifact': 'Artifact'
      };
      
      return valueMap[lastPart] || lastPart;
    }
    
    if (value.includes('_g_L') || value.match(/_[a-z]_L\d+C\d+/)) {
      return 'No Security';
    }
    
    return value;
  };
  
  export const getPropertyType = (predicateUri) => {
    if (predicateUri.includes('hasSecurityConfiguration')) {
      return 'security';
    }
    if (predicateUri.includes('isContainedIn')) {
      return 'container';
    }
    return 'standard';
  };
  