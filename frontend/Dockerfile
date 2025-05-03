# FILE: frontend/Dockerfile

# Stage 1: Build the React application
FROM node:18-alpine as build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --legacy-peer-deps
COPY . .
RUN npm run build

# Stage 2: Serve the build output with Nginx
FROM nginx:1.25-alpine
# Copy built assets from the build stage
COPY --from=build /app/build /usr/share/nginx/html
# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]