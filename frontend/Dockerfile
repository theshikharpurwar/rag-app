# FILE: frontend/Dockerfile
# (Workspace-Aware Version - CORRECT CODE)

# Stage 1: Build the React application using Node 20+
FROM node:20-alpine as build

# Set base working directory (represents project root in this stage)
WORKDIR /app

# Copy the root package.json AND the root package-lock.json
COPY package.json ./
COPY package-lock.json ./

# Copy the package.json specific to the frontend workspace
COPY frontend/package.json ./frontend/

# Install ONLY the frontend workspace dependencies using the root lock file
RUN npm ci --workspace=frontend --legacy-peer-deps

# Copy the rest of the frontend source code into its subdirectory
COPY frontend ./frontend/

# Set the working directory to the frontend code for the build step
WORKDIR /app/frontend

# Run the build script defined in frontend/package.json
RUN npm run build

# Stage 2: Serve the build output with Nginx
FROM nginx:1.25-alpine
# Copy built assets from the build stage's frontend/build directory
COPY --from=build /app/frontend/build /usr/share/nginx/html
# Optional: Copy custom Nginx config if needed
# COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]