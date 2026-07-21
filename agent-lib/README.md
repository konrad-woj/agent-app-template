# Agent library

## Quick example

Tun postgresql database

```bash
docker run --name langgraph-db \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=langgraph \
  -e POSTGRES_USER=postgres \
  -p 5433:5432 \
  -d postgres:17
```

Run the agent library

```bash
cd agent-lib
uv sync
uv run python -m agent_lib.main
```

