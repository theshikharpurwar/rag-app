# FILE: frontend/Dockerfile

# Stage 1: Build the React application
FROM node:18-alpine AS build
WORKDIR /app

# Set build-time environment variables
ENV REACT_APP_API_URL=http://localhost:5000/api
ENV CI=false

COPY package.json ./
# Use npm install instead of npm ci for better compatibility
RUN npm install --legacy-peer-deps
COPY . .
# Add --verbose to see what's failing in the build
RUN npm run build --verbose || (echo "Build failed, showing package.json:" && cat package.json && exit 1)

# Stage 2: Serve the build output with Nginx
FROM nginx:1.25-alpine
# Copy built assets from the build stage
COPY --from=build /app/build /usr/share/nginx/html
# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]