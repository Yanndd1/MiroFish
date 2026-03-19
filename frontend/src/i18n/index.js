import { createI18n } from 'vue-i18n'
import zh from './locales/zh.json'
import en from './locales/en.json'
import fr from './locales/fr.json'

const i18n = createI18n({
  legacy: false,
  locale: localStorage.getItem('locale') || 'zh',
  fallbackLocale: 'en',
  messages: { zh, en, fr }
})

export default i18n
