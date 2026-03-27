import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: false,
  // Suppress dev overlay for browser-extension hydration noise (bis_skin_checked)
  devIndicators: false,
  async rewrites() {
    return [
      { source: '/sessions/:path*', destination: 'http://localhost:8000/sessions/:path*' },
      { source: '/upload', destination: 'http://localhost:8000/upload' },
      { source: '/analyze', destination: 'http://localhost:8000/analyze' },
      { source: '/agent/:path*', destination: 'http://localhost:8000/agent/:path*' },
      { source: '/kernel/:path*', destination: 'http://localhost:8000/kernel/:path*' },
      { source: '/files/:path*', destination: 'http://localhost:8000/files/:path*' },
      { source: '/save-results', destination: 'http://localhost:8000/save-results' },
      { source: '/api/:path*', destination: 'http://localhost:8000/:path*' }
    ];
  }
};

export default nextConfig;
