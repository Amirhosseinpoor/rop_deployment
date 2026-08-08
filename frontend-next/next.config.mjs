/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  // Assets are served by Django from /static/landing_next/_next/...
  assetPrefix: '/static/landing_next',
  images: { unoptimized: true },
  reactStrictMode: false,
};

export default nextConfig;
