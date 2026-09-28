export async function POST(request: Request) {
  const apiUrl = process.env.API_INTERNAL_URL ?? "http://127.0.0.1:8001";
  const upstreamResponse = await fetch(`${apiUrl}/api/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": request.headers.get("content-type") ?? "application/json",
      Accept: "text/event-stream",
    },
    body: await request.text(),
    cache: "no-store",
  });

  return new Response(upstreamResponse.body, {
    status: upstreamResponse.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
