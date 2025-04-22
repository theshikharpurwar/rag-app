# FILE: frontend/Dockerfile
# (Updated Node version)

# Stage 1: Build the React application using Node 20
FROM node:20-alpine as build 
WORKDIR /app
COPY package.json ./
COPY package-lock.json ./ 
# Use npm ci for clean install based on lockfile
RUN npm ci --legacy-peer-deps
COPY . .
RUN npm run build

# Stage 2: Serve the build output with Nginx
FROM nginx:1.25-alpine
COPY --from=build /app/build /usr/share/nginx/html
# Optional: Copy custom Nginx config if needed for routing
# COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]