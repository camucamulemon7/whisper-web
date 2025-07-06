FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package.json ./
RUN npm install

# Copy source code
COPY . .

# Expose Vite port
EXPOSE 5173

# Run development server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
