FROM gcc:14-bookworm AS build
COPY src/shop.cpp /src/shop.cpp
RUN g++ -O3 -DNDEBUG -std=c++17 -pthread -static-libstdc++ -static-libgcc /src/shop.cpp -o /shop

FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir polars==1.35.2 && useradd -u 1000 -M -d /scratch app
COPY --from=build /shop /app/shop
COPY src/write.py run.sh /app/
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
USER 1000:1000
ENTRYPOINT ["/bin/sh", "/app/run.sh"]
