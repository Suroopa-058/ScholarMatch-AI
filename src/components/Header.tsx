import { useState } from "react";
import { Menu, X } from "lucide-react";
import { NavLink } from "react-router-dom";
import { getAuth } from "../utils/auth";

const links = [{ label: "Home", to: "/" }, { label: "Find Scholarships", to: "/scholarships" }, { label: "My Profile", to: "/profile" }, { label: "About", to: "/about" }];

export default function Header() {
  const [open, setOpen] = useState(false);
  const user = getAuth()?.user;
  const linkClass = ({ isActive }: { isActive: boolean }) => `text-sm font-medium transition-colors hover:text-[var(--color-navy)] ${isActive ? "text-[var(--color-navy)]" : "text-[var(--color-muted)]"}`;
  return <header className="sticky top-0 z-50 border-b border-[var(--color-line)] bg-[var(--color-paper)]/95 backdrop-blur"><div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3.5 sm:px-6"><NavLink to="/" className="flex items-center gap-2.5" aria-label="ScholarMatch AI home"><SealMark /><span className="font-display text-[1.05rem] font-semibold tracking-tight text-[var(--color-navy)]">ScholarMatch <span className="text-[var(--color-brass)]">AI</span></span></NavLink><nav className="hidden items-center gap-7 md:flex">{links.map((link) => <NavLink key={link.to} to={link.to} className={linkClass}>{link.label}</NavLink>)}</nav>{user ? <NavLink to="/profile" className="hidden rounded-full bg-[var(--color-navy)] px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-[var(--color-navy-deep)] md:block">Hi, {user.name.split(" ")[0]}</NavLink> : <NavLink to="/login" className="hidden rounded-full bg-[var(--color-navy)] px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-[var(--color-navy-deep)] md:block">Sign in</NavLink>}<button className="rounded p-1 md:hidden" aria-label={open ? "Close navigation menu" : "Open navigation menu"} aria-expanded={open} onClick={() => setOpen(!open)}>{open ? <X size={22} /> : <Menu size={22} />}</button></div>{open && <nav className="border-t border-[var(--color-line)] bg-white px-6 py-4 md:hidden">{links.map((link) => <NavLink key={link.to} to={link.to} onClick={() => setOpen(false)} className="block py-2.5 text-sm font-medium text-[var(--color-muted)]">{link.label}</NavLink>)}<NavLink to={user ? "/profile" : "/login"} onClick={() => setOpen(false)} className="mt-2 block rounded-full bg-[var(--color-navy)] px-5 py-2.5 text-center text-sm font-semibold text-white">{user ? "My Profile" : "Sign in"}</NavLink></nav>}</header>;
}

export function SealMark({ size = 30 }: { size?: number }) { return <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true"><circle cx="32" cy="32" r="30" fill="var(--color-navy)" /><circle cx="32" cy="32" r="24" fill="none" stroke="var(--color-brass)" strokeWidth="1.5" /><path d="M32 18 L44 24 L32 30 L20 24 Z" fill="var(--color-brass)" /><path d="M22 27 V35 C22 38 27 40 32 40 C37 40 42 38 42 35 V27" fill="none" stroke="var(--color-paper)" strokeWidth="1.6" /></svg>; }
