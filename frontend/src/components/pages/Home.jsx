import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="hero min-h-100">
      <div className="hero-content text-center">
        <div className="max-w-lg px-4">
          {/* Responsive Heading */}
          <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold text-primary">
            Voicefaux
          </h1>

          {/* Responsive Paragraph */}
          <p className="py-6 sm:py-8 text-base sm:text-lg md:text-xl lg:text-2xl">
              Detect whether an audio recording is authentic or manipulated.
            Voicefaux analyzes voice samples to identify unmodified,
            synthetic, modified, or spliced audio with confidence.
          </p>

          {/* CTA Button */}
          <Link
            to="/classify"
            className="btn btn-primary btn-lg"
          >
             Analyze Audio
          </Link>
        </div>
      </div>
    </div>
  );
}
