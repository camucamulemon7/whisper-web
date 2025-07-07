#!/bin/bash

docker compose build --no-cache
docker compose -f docker-compose.yaml --force-recreate --detach
