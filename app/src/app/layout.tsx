import "./globals.css";

import type { Metadata } from "next";
import { Inter as FontSans } from "next/font/google";

import { cn } from "@/lib/utils";

export const metadata: Metadata = {
  title: "DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research",
  description: "DR Tulu-8B is the first open deep research model directly trained for open-ended, long-form research using Reinforcement Learning with Evolving Rubrics (RLER).",
};

const fontSans = FontSans({
  subsets: ["latin"],
  variable: "--font-sans",
});

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        className={cn(
          "min-h-screen bg-background font-sans antialiased",
          fontSans.variable
        )}
      >
        {children}
      </body>
    </html>
  );
}
