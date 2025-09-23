"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";

type Side = "left" | "right";

type Props = {
  initialWidth?: number; // default 480
  minWidth?: number;     // default 320
  maxWidth?: number;     // default 900
  side?: Side;           // default "right"
  className?: string;
  children: React.ReactNode;
};

export default function DraggableSidePanel({
  initialWidth = 480,
  minWidth = 320,
  maxWidth = 900,
  side = "right",
  className,
  children,
}: Props) {
  const [width, setWidth] = useState(() =>
    Math.min(Math.max(initialWidth, minWidth), maxWidth)
  );
  const draggingRef = useRef(false);

  // Refs to store latest bounds without re-binding handlers
  const minRef = useRef(minWidth);
  const maxRef = useRef(maxWidth);
  const sideRef = useRef<Side>(side);
  useEffect(() => { minRef.current = minWidth; }, [minWidth]);
  useEffect(() => { maxRef.current = maxWidth; }, [maxWidth]);
  useEffect(() => { sideRef.current = side; }, [side]);

 const handleDown = useCallback(
  (e: React.MouseEvent<HTMLDivElement> | React.TouchEvent<HTMLDivElement>) => {
    e.preventDefault();
    draggingRef.current = true;

    const onMove = (ev: MouseEvent | TouchEvent) => {
      if (!draggingRef.current) return;

      let clientX: number;
      if (ev instanceof TouchEvent) {
        clientX = ev.touches[0]?.clientX ?? 0;
      } else {
        clientX = (ev as MouseEvent).clientX;
      }

      let newWidth: number;
      const w = window.innerWidth;
      if (sideRef.current === "right") {
        // Divider is on the LEFT edge of the right panel
        newWidth = w - clientX; // distance from right edge
      } else {
        // Left side: divider on RIGHT edge of left panel
        newWidth = clientX; // distance from left edge
      }

      newWidth = Math.max(minRef.current, Math.min(maxRef.current, newWidth));
      setWidth(newWidth);
      document.body.style.cursor = "col-resize";
    };

    const onUp = () => {
      draggingRef.current = false;
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.removeEventListener("touchmove", onMove);
      document.removeEventListener("touchend", onUp);
      document.body.style.cursor = "";
    };

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    document.addEventListener("touchmove", onMove, { passive: false });
    document.addEventListener("touchend", onUp);
  },
  []
);


  const layout = useMemo(() => {
    const Handle = (
      <div
        role="separator"
        aria-orientation="vertical"
        title="Drag to resize"
        onMouseDown={handleDown}
        onTouchStart={handleDown}
        className={clsx(
          "relative z-20 h-full w-[8px] shrink-0 cursor-col-resize",
          "bg-neutral-200 hover:bg-neutral-300 active:bg-neutral-300",
          // a slim inner grip for visibility
          "group"
        )}
      >
        {/* Grip dots */}
        <div
          className={clsx(
            "absolute top-1/2 -translate-y-1/2",
            // place dots centered inside the 8px rail
            "left-1/2 -translate-x-1/2",
            "flex flex-col gap-1"
          )}
        >
          <span className="block h-1 w-1 rounded-full bg-neutral-500/60 group-hover:bg-neutral-700/70" />
          <span className="block h-1 w-1 rounded-full bg-neutral-500/60 group-hover:bg-neutral-700/70" />
          <span className="block h-1 w-1 rounded-full bg-neutral-500/60 group-hover:bg-neutral-700/70" />
        </div>
      </div>
    );

    const Panel = (
      <div
        className={clsx(
          "h-full overflow-auto bg-white shadow-sm border-l border-neutral-200",
          side === "left" && "border-l-0 border-r",
          className
        )}
        style={{ width }}
      >
        {children}
      </div>
    );

    if (side === "right") {
      // [spacer] [HANDLE (on left edge of panel)] [PANEL]
      return (
        <>
          <div className="flex-1 h-full" />
          {Handle}
          {Panel}
        </>
      );
    }
    // side === "left"
    // [PANEL] [HANDLE (on right edge of panel)] [spacer]
    return (
      <>
        {Panel}
        {Handle}
        <div className="flex-1 h-full" />
      </>
    );
  }, [children, className, handleDown, side, width]);

  return (
    <div className="w-full h-screen flex overflow-hidden bg-neutral-50">
      {layout}
    </div>
  );
}
