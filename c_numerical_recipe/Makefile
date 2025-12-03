CC      := gcc
# -Iinclude tells the compiler to look in the 'include' folder for headers
CFLAGS := -Wall -Wextra -std=c99 -Isrc

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

TARGET  := $(BIN_DIR)/myapp
SOURCES := $(wildcard $(SRC_DIR)/*.c)
OBJECTS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SOURCES))

.PHONY: all clean

all: $(TARGET)

# Linking Rule
$(TARGET): $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(LDFLAGS) $^ -o $@

# Compilation Rule
# Note the pattern: obj/%.o depends on src/%.c
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)