import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
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
  title: "AI Data Analyst",
  description: "Multi-agent hierarchical data analysis platform powered by CrewAI and Gemini",
};

/**
 * Synchronous inline script that patches console.error BEFORE React hydration.
 * This must run before Next.js DevOverlay captures the error.
 * It suppresses the false-positive hydration mismatch caused by browser
 * security extensions (McAfee/Norton/Bitdefender BIS) injecting
 * `bis_skin_checked` attributes into every <div>.
 */
const HYDRATION_SUPPRESS_SCRIPT = `
(function(){
  var _oe = console.error;
  console.error = function(){
    var a = arguments;
    for(var i = 0; i < a.length; i++){
      if(typeof a[i] === 'string' && (
        a[i].indexOf('bis_skin_checked') !== -1 ||
        a[i].indexOf('bis_register') !== -1 ||
        a[i].indexOf('__processed_') !== -1
      )) return;
    }
    return _oe.apply(console, a);
  };
})();
`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <head>
        {/* Must execute synchronously before React hydration starts */}
        <script dangerouslySetInnerHTML={{ __html: HYDRATION_SUPPRESS_SCRIPT }} />
      </head>
      <body className="min-h-full flex flex-col" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
