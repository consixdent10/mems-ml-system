# Frontend Dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Build for production
RUN npm run build

# Expose port
EXPOSE 5173

# Run development server (for demo) or use 'serve' for production
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0"]
