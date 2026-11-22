FROM gcc:14-bookworm AS build
COPY src/shop.cpp src/packed.hpp src/parquet.hpp /src/
RUN g++ -O3 -DNDEBUG -std=c++17 -pthread -static /src/shop.cpp \
    -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -o /shop

FROM scratch
COPY --from=build /shop /shop
USER 1000:1000
ENTRYPOINT ["/shop"]
