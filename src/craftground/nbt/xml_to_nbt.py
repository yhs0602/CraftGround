import xml.etree.ElementTree as ET

from structure_editor import Structure


def parse_drawing_decorator(xml_file: str, output_nbt: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    namespace = {"malmo": "http://ProjectMalmo.microsoft.com"}

    structure = Structure()

    # Find the DrawingDecorator section
    drawing_decorator = root.find(".//malmo:DrawingDecorator", namespace)
    if drawing_decorator is None:
        raise ValueError("No DrawingDecorator found in XML.")

    # Process elements in order to preserve drawing priority
    for element in drawing_decorator:
        tag = element.tag.split("}")[-1]  # Remove namespace prefix

        if tag == "DrawCuboid":
            block_type = element.get("type")
            x1, y1, z1 = (
                int(element.get("x1")),
                int(element.get("y1")),
                int(element.get("z1")),
            )
            x2, y2, z2 = (
                int(element.get("x2")),
                int(element.get("y2")),
                int(element.get("z2")),
            )
            structure.set_cuboid(x1, y1, z1, x2, y2, z2, f"minecraft:{block_type}")

        elif tag == "DrawBlock":
            block_type = element.get("type")
            x, y, z = (
                int(element.get("x")),
                int(element.get("y")),
                int(element.get("z")),
            )
            structure.set_block(x, y, z, f"minecraft:{block_type}")

    # Save the parsed structure as an NBT file
    structure.save(output_nbt)
    print(f"Structure saved to {output_nbt}")


# Example usage
if __name__ == "__main__":
    parse_drawing_decorator("nbts/mission.xml", "nbts/mission.nbt")
