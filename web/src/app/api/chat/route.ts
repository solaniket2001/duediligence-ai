export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const apiUrl = process.env.API_INTERNAL_URL ?? "http://127.0.0.1:8001";
  const upstreamResponse = await fetch(`${apiUrl}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: await request.text(),
    cache: "no-store",
  });

  const data = await upstreamResponse.text();
  return new Response(data, {
    status: upstreamResponse.status,
    headers: { "Content-Type": "application/json" },
  });
}