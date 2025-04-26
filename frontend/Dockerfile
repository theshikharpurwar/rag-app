# FILE: frontend/Dockerfile
# (Workspace-Aware - Install All Deps - Run build from frontend dir with relative path script)

# Stage 1: Build the React application using Node 20+
FROM node:20-alpine as build

WORKDIR /app # Root context

# Copy the root package.json AND the root package-lock.json
COPY package.json ./
COPY package-lock.json ./

# Install ALL dependencies using the root lockfile first
# This ensures react-scripts is in /app/node_modules/.bin
RUN npm ci --legacy-peer-deps

# Copy the rest of the frontend source code into its subdirectory
COPY frontend ./frontend/

# Set the working directory to the frontend code
WORKDIR /app/frontend

# Run the build script (defined in frontend/package.json, which now uses relative path)
RUN npm run build

# Stage 2: Serve the build output with Nginx
FROM nginx:1.25-alpine
# Copy built assets from the build stage's frontend/build directory
COPY --from=build /app/frontend/build /usr/share/nginx/html
# Optional: Copy custom Nginx config if needed
# COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]