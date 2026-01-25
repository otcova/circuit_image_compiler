## UI

- Display net id as a clickable (inline button) that lets user hover and select such net.
- CollaspsableHeaders aligned with content.
- Measure and show Parsing and Engine Compilation times
- Do not force repaint when idle
- Icons, (Folder image?)

## Engine Optimization

- Gates without wires or with only the ON wire connected as output wire and/or control.

- Preprocess Union Find graph to find a set number of aliases that can be removed to not produce cycles.
- Process Union Find separating completely trivial wires.

- Dfs real stack (not todo list)
- Dfs both visited map for wire nodes and gate nodes might not be needed

## Others

- Make the backend track simulation states every few ms,
  then we can jumb backwards in time by simulating from closest point.
