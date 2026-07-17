import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Architect SaaS — Floor Plan to 3D",
  description: "Convert 2D architectural floor plans into BIM-ready 3D models.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-neutral-950 text-neutral-100 antialiased">
        {children}
      </body>
    </html>
  );
}
