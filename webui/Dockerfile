# syntax=docker/dockerfile:1.7

# ---- Builder ----
FROM node:20-alpine AS builder
WORKDIR /app

# NEXT_PUBLIC_* must be present at build time (inlined into the client bundle).
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ARG NEXT_PUBLIC_API_KEY=
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
ENV NEXT_PUBLIC_API_KEY=$NEXT_PUBLIC_API_KEY

# Install dependencies using the lockfile for reproducible builds.
COPY package.json package-lock.json* ./
RUN npm ci

# Copy source and build the standalone Next.js output.
COPY . .
RUN npm run build

# ---- Runner ----
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0
# Echo build-time public URL for operators (client already baked).
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL

# Run as non-root user.
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy the standalone output, static assets and public folder.
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs

EXPOSE 3000

CMD ["node", "server.js"]
