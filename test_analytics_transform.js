// Test analytics data transformation
const mockHealthData = {
  "status": "healthy",
  "timestamp": "2025-07-25T13:30:00Z",
  "components": {
    "system_resources": {
      "status": "healthy",
      "details": {
        "cpu_usage_percent": 45.2,
        "memory_usage_percent": 62.8,
        "memory_available_gb": 3.2
      }
    },
    "disk_usage": {
      "status": "healthy", 
      "details": {
        "total_gb": 500.0,
        "used_gb": 320.5,
        "free_gb": 179.5,
        "usage_percent": 64.1
      }
    }
  }
};

// Transform the health data format (mimicking frontend code)
const components = Object.entries(mockHealthData.components || {}).map(([name, component]) => ({
  name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
  status: component.status || 'healthy',
  details: component.details || {},
  type: name
}));

const result = {
  status: mockHealthData.status || 'healthy',
  components
};

console.log("Transformed health data:");
console.log(JSON.stringify(result, null, 2));

console.log("\nComponent names and types:");
result.components.forEach(comp => {
  console.log(`- ${comp.name} (type: ${comp.type})`);
  console.log(`  Status: ${comp.status}`);
  console.log(`  Details:`, comp.details);
});