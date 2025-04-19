# FILE: frontend/Dockerfile

# Stage 1: Build the React application
FROM node:18-alpine as build
WORKDIR /app
COPY package.json ./
COPY package-lock.json* ./
# Use ci --legacy-peer-deps if needed based on your project's dependencies
RUN npm ci --legacy-peer-deps
COPY . .
RUN npm run build

# Stage 2: Serve the build output with Nginx
FROM nginx:1.25-alpine
COPY --from=build /app/build /usr/share/nginx/html
# Optional: Copy custom Nginx config for single-page app routing
# COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]