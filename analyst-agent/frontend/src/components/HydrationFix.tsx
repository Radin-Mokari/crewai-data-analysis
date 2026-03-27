"use client";

/**
 * HydrationFix — runs on the client only.
 *
 * Suppresses the spurious hydration warning caused by browser security
 * extensions (McAfee, Norton, Bitdefender BIS suite) that inject attributes
 * like `bis_skin_checked`, `bis_register`, `__processed_*` into every <div>
 * — including Next.js internal elements we cannot add suppressHydrationWarning to.
 *
 * This is NOT suppressing real React errors; only these known false-positives.
 */
import { useEffect } from "react";

export default function HydrationFix() {
  useEffect(() => {
    const orig = console.error.bind(console);
    console.error = (...args: unknown[]) => {
      const msg = typeof args[0] === "string" ? args[0] : "";
      if (
        msg.includes("bis_skin_checked") ||
        msg.includes("bis_register") ||
        msg.includes("__processed_") ||
        msg.includes("data-testim") ||
        // The generic hydration mismatch message when caused by BIS
        (msg.includes("hydrated") && args.some((a) =>
          typeof a === "string" && (a.includes("bis_") || a.includes("__processed_"))
        ))
      ) {
        return; // swallow browser-extension noise
      }
      orig(...args);
    };

    return () => {
      // Restore original on unmount (only relevant for HMR in dev)
      console.error = orig;
    };
  }, []);

  return null; // renders nothing
}
