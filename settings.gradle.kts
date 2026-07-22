// IDE-only composite workspace. Not used by CI or by minecraft/mc121's own build —
// each Minecraft version keeps its own Gradle wrapper/Loom version (see docs/26_2_work_plan.md).
rootProject.name = "craftground-workspace"

includeBuild("minecraft/mc121")
