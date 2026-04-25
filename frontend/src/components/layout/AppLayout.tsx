import { Outlet } from "react-router-dom";
import { SiteHeader } from "@/components/layout/SiteHeader";

export function AppLayout() {
  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />
      <Outlet />
    </div>
  );
}
