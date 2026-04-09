import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { ThemeProvider } from "@/components/theme-provider";
import { Navbar } from "@/components/layout/Navbar";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "BookRec — ML Recommender System",
  description:
    "An interactive book recommender system built with Collaborative Filtering, Content-Based, and Hybrid algorithms. Portfolio project by a machine learning engineer.",
  openGraph: {
    title: "BookRec — ML Recommender System",
    description:
      "Interactive demo of CF, Content-Based, and Hybrid recommendation algorithms on real book data.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} dark`}
      suppressHydrationWarning
    >
      <body className="min-h-screen bg-zinc-950 font-sans text-zinc-100 antialiased">
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
          <TooltipProvider>
            <Navbar />
            <main className="mx-auto max-w-6xl px-6 py-10">{children}</main>
            <footer className="border-t border-zinc-800 py-8 text-center text-sm text-zinc-500">
              Built with Python · scikit-learn · FastAPI · Next.js 16
            </footer>
          </TooltipProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
