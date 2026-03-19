<template>
  <div class="lang-switcher" :class="{ open: isOpen }" ref="switcherRef">
    <button class="lang-btn" @click="isOpen = !isOpen" :title="$t('common.language') || 'Language'">
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5">
        <circle cx="12" cy="12" r="10"></circle>
        <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
      </svg>
      <span class="lang-code">{{ currentLang.code }}</span>
    </button>
    <Transition name="dropdown">
      <div v-if="isOpen" class="lang-dropdown">
        <button
          v-for="lang in languages"
          :key="lang.value"
          class="lang-option"
          :class="{ active: locale === lang.value }"
          @click="switchLocale(lang.value)"
        >
          <span class="lang-flag">{{ lang.flag }}</span>
          <span class="lang-name">{{ lang.label }}</span>
        </button>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'

const { locale } = useI18n()
const isOpen = ref(false)
const switcherRef = ref(null)

const languages = [
  { value: 'zh', label: '中文', code: 'ZH', flag: '🇨🇳' },
  { value: 'en', label: 'English', code: 'EN', flag: '🇬🇧' },
  { value: 'fr', label: 'Français', code: 'FR', flag: '🇫🇷' }
]

const currentLang = computed(() => {
  return languages.find(l => l.value === locale.value) || languages[0]
})

const switchLocale = (val) => {
  locale.value = val
  localStorage.setItem('locale', val)
  isOpen.value = false
}

const handleClickOutside = (e) => {
  if (switcherRef.value && !switcherRef.value.contains(e.target)) {
    isOpen.value = false
  }
}

onMounted(() => document.addEventListener('click', handleClickOutside))
onUnmounted(() => document.removeEventListener('click', handleClickOutside))
</script>

<style scoped>
.lang-switcher {
  position: relative;
  z-index: 1000;
}

.lang-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  background: transparent;
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 6px;
  cursor: pointer;
  color: inherit;
  font-size: 12px;
  font-weight: 500;
  font-family: 'Space Grotesk', monospace;
  transition: all 0.2s;
}

.lang-btn:hover {
  background: rgba(0, 0, 0, 0.04);
  border-color: rgba(0, 0, 0, 0.3);
}

.lang-code {
  letter-spacing: 0.5px;
}

.lang-dropdown {
  position: absolute;
  top: calc(100% + 4px);
  right: 0;
  background: #fff;
  border: 1px solid rgba(0, 0, 0, 0.1);
  border-radius: 8px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  min-width: 140px;
}

.lang-option {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 8px 12px;
  background: none;
  border: none;
  cursor: pointer;
  font-size: 13px;
  font-family: inherit;
  color: #333;
  transition: background 0.15s;
}

.lang-option:hover {
  background: rgba(0, 0, 0, 0.04);
}

.lang-option.active {
  background: rgba(0, 0, 0, 0.06);
  font-weight: 600;
}

.lang-flag {
  font-size: 16px;
}

.lang-name {
  flex: 1;
  text-align: left;
}

.dropdown-enter-active,
.dropdown-leave-active {
  transition: all 0.15s ease;
}

.dropdown-enter-from,
.dropdown-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>
