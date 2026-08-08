import type { Metadata } from "next";
import { Sora, Manrope, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const display = Sora({ subsets: ["latin"], weight: ["600", "700", "800"], variable: "--f-display" });
const body = Manrope({ subsets: ["latin"], weight: ["400", "500", "600", "700"], variable: "--f-body" });
const mono = JetBrains_Mono({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--f-mono" });

export const metadata: Metadata = {
  title: "Mediverse AI — Clinical Intelligence for Eyes & Whole-Patient Health",
  description:
    "Mediverse AI reads retinal scans, corneal maps, lab panels and workplace exposures, then returns cited, doctor-grade diagnostics and work-fitness decisions.",
  icons: { icon: "/static/icons/mediversai_logo_final-032.png" },
};

export const viewport = { themeColor: "#05060c", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
