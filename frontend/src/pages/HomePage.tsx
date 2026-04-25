import { Link } from "react-router-dom";
import {
  ArrowRight,
  Building2,
  Check,
  LineChart,
  Shield,
  Sparkles,
  Users,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const plans = [
  {
    name: "Starter",
    price: "Free",
    blurb: "Try the full research pipeline on tickers you care about.",
    features: ["Equity notes with PDF preview", "Headline scan + filings context", "Library storage for your runs"],
    cta: "Create free account",
    href: "/register",
    highlight: false,
  },
  {
    name: "Pro",
    price: "$29",
    period: "/ mo",
    blurb: "For active investors who want a faster rhythm and priority access as we ship it.",
    features: ["Everything in Starter", "Higher usage limits (coming soon)", "Email summaries (coming soon)"],
    cta: "Start with Starter",
    href: "/register",
    highlight: true,
  },
  {
    name: "Institution",
    price: "Custom",
    blurb: "Bring your own data feeds, compliance workflows, and dedicated support.",
    features: ["Private data connectors", "SSO & audit trails (roadmap)", "Custom models & retention"],
    cta: "Talk to us",
    href: "mailto:hello@example.com",
    highlight: false,
    external: true,
  },
];

export function HomePage() {
  return (
    <div className="flex-1 bg-[radial-gradient(ellipse_at_top,_hsl(152_76%_42%/0.14),_transparent_50%),radial-gradient(ellipse_at_bottom,_hsl(215_28%_14%/0.92),_hsl(224_71%_4%))]">
      <section className="mx-auto max-w-6xl px-4 pb-16 pt-12 sm:pt-20">
        <div className="mx-auto max-w-3xl text-center">
          <p className="font-mono text-xs uppercase tracking-[0.22em] text-primary/90">
            B2C research · B2B when you need scale
          </p>
          <h1 className="mt-4 text-balance text-4xl font-semibold tracking-tight text-foreground sm:text-5xl">
            Institutional-grade equity notes, built for individuals
          </h1>
          <p className="mt-5 text-pretty text-lg text-muted-foreground sm:text-xl">
            Sentinel Research pulls filings, market context, and a disciplined narrative pass into a
            single report you can preview, download, and revisit anytime in your personal library.
          </p>
          <div className="mt-10 flex flex-wrap items-center justify-center gap-3">
            <Button size="lg" className="gap-2 px-8" asChild>
              <Link to="/research">
                <LineChart className="h-4 w-4" />
                Open research workspace
                <ArrowRight className="h-4 w-4 opacity-80" />
              </Link>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <Link to="/register">Create account</Link>
            </Button>
          </div>
          <p className="mt-4 text-xs text-muted-foreground">
            Not investment advice. Models and data can be wrong — verify before acting.
          </p>
        </div>

        <div className="mx-auto mt-20 grid max-w-5xl gap-6 sm:grid-cols-3">
          <div className="rounded-xl border border-border/80 bg-card/50 p-6 backdrop-blur-sm">
            <Sparkles className="h-8 w-8 text-primary" aria-hidden />
            <h2 className="mt-4 text-lg font-semibold text-foreground">Multi-agent pipeline</h2>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              Junior researcher, critic, and lead PM passes produce a structured memo and print-ready
              LaTeX in one flow.
            </p>
          </div>
          <div className="rounded-xl border border-border/80 bg-card/50 p-6 backdrop-blur-sm">
            <Shield className="h-8 w-8 text-primary" aria-hidden />
            <h2 className="mt-4 text-lg font-semibold text-foreground">Transparent sources</h2>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              Filing facts, market history, and headline scans are surfaced so you can trace the story
              back to primary material.
            </p>
          </div>
          <div className="rounded-xl border border-border/80 bg-card/50 p-6 backdrop-blur-sm">
            <Users className="h-8 w-8 text-primary" aria-hidden />
            <h2 className="mt-4 text-lg font-semibold text-foreground">Your library</h2>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              Signed-in users get every completed report saved automatically—pick up where you left off
              on any device.
            </p>
          </div>
        </div>
      </section>

      <section className="border-t border-border/60 bg-card/20 py-16 backdrop-blur-sm">
        <div className="mx-auto max-w-6xl px-4">
          <div className="mx-auto max-w-2xl text-center">
            <Building2 className="mx-auto h-8 w-8 text-primary" aria-hidden />
            <h2 className="mt-4 text-3xl font-semibold tracking-tight text-foreground">Plans</h2>
            <p className="mt-3 text-muted-foreground">
              Pricing is indicative while we finish metering. Today, accounts are free and reports save
              to your library when you are logged in.
            </p>
          </div>
          <div className="mt-12 grid gap-6 lg:grid-cols-3">
            {plans.map((p) => (
              <Card
                key={p.name}
                className={`flex flex-col border-border/80 bg-card/70 ${
                  p.highlight ? "border-primary/50 shadow-lg shadow-primary/10 ring-1 ring-primary/25" : ""
                }`}
              >
                <CardHeader>
                  <CardTitle className="text-xl">{p.name}</CardTitle>
                  <CardDescription className="text-base text-muted-foreground">{p.blurb}</CardDescription>
                  <p className="pt-2 text-3xl font-semibold tracking-tight text-foreground">
                    {p.price}
                    {"period" in p && p.period ? (
                      <span className="text-lg font-normal text-muted-foreground">{p.period}</span>
                    ) : null}
                  </p>
                </CardHeader>
                <CardContent className="flex-1">
                  <ul className="space-y-2">
                    {p.features.map((f) => (
                      <li key={f} className="flex gap-2 text-sm text-muted-foreground">
                        <Check className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden />
                        <span>{f}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
                <CardFooter>
                  {"external" in p && p.external ? (
                    <Button className="w-full" variant="outline" asChild>
                      <a href={p.href}>{p.cta}</a>
                    </Button>
                  ) : (
                    <Button
                      className="w-full"
                      variant={p.highlight ? "default" : "outline"}
                      asChild
                    >
                      <Link to={p.href}>{p.cta}</Link>
                    </Button>
                  )}
                </CardFooter>
              </Card>
            ))}
          </div>
        </div>
      </section>

      <footer className="border-t border-border/60 py-10 text-center text-xs text-muted-foreground">
        <p>Sentinel Research — consumer-first equity tooling with an institutional lane.</p>
      </footer>
    </div>
  );
}
